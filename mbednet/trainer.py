import os
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .model import MbEdNet
from .loss import MbEdLoss
from .metrics import postprocess_segmentation, macro_dice


class ModelEMA:
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_tensor(v) and v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(
                    v.detach().to(self.shadow[k].dtype), alpha=1.0 - self.decay)


class EMASwap:
    def __init__(self, ema, model):
        self.ema = ema
        self.model = model
        self.backup = None

    def __enter__(self):
        sd = self.model.state_dict()
        self.backup = {k: sd[k].detach().clone() for k in self.ema.shadow}
        for k, v in self.ema.shadow.items():
            sd[k].copy_(v)
        return self

    def __exit__(self, *args):
        sd = self.model.state_dict()
        for k, v in self.backup.items():
            sd[k].copy_(v)
        self.backup = None


def enable_gpu_optimizations():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


class MbEdTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        self.model = MbEdNet(
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            features=config.features,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            elrs_rank=config.elrs_rank,
        ).to(self.device)
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")

        self.criterion = MbEdLoss(
            num_classes=config.num_classes, edge_lambda=config.edge_lambda).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scaler = GradScaler(enabled=config.use_amp)
        self.ema = ModelEMA(self.model, decay=0.99)

        self.best_macro = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.start_epoch = 0
        self._accum_n = max(1, int(config.grad_accum_steps))
        self._accum_step = 0
        self.nan_batch_count = 0

        self._resume()

        if self.start_epoch > 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=1e-6, last_epoch=self.start_epoch - 1)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=1e-6)
        for g, lr in zip(self.optimizer.param_groups, self.scheduler.get_last_lr()):
            g["lr"] = lr

    def _ckpt_path(self, name):
        return os.path.join(self.config.checkpoint_dir, name)

    def _resume(self):
        for name in ("checkpoint_latest.pth", "best_checkpoint.pth"):
            path = self._ckpt_path(name)
            if not os.path.exists(path):
                continue
            ck = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ck["model_state_dict"], strict=False)
            if "optimizer_state_dict" in ck and ck["optimizer_state_dict"]:
                try:
                    self.optimizer.load_state_dict(ck["optimizer_state_dict"])
                except Exception:
                    pass
            if "scaler_state_dict" in ck:
                try:
                    self.scaler.load_state_dict(ck["scaler_state_dict"])
                except Exception:
                    pass
            if "ema_shadow" in ck:
                self.ema.shadow = {k: v.to(self.device) for k, v in ck["ema_shadow"].items()}
            self.start_epoch = int(ck.get("epoch", -1)) + 1
            self.best_macro = float(ck.get("best_macro", 0.0))
            self.best_epoch = int(ck.get("best_epoch", 0))
            self.epochs_without_improvement = int(ck.get("epochs_without_improvement", 0))
            self.criterion.alpha = float(ck.get("alpha", self.criterion.alpha))
            print(f"Resumed from {path} at epoch {self.start_epoch}")
            return

    def _save(self, epoch, name, is_best=False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "ema_shadow": self.ema.shadow,
            "alpha": self.criterion.alpha,
            "best_macro": self.best_macro,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
        }
        torch.save(state, self._ckpt_path(name))

    def _train_step(self, batch):
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        dwm_t = batch["dwm"].to(self.device, non_blocking=True)

        if self._accum_step == 0:
            self.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=self.config.use_amp, dtype=torch.bfloat16):
            loss, loss_dict = self._forward_loss(images, labels, dwm_t)

        if not torch.isfinite(loss):
            with autocast(enabled=False):
                loss, loss_dict = self._forward_loss(images.float(), labels, dwm_t)

        if not torch.isfinite(loss):
            self.nan_batch_count += 1
            return torch.tensor(0.0, device=self.device), {"total": 0.0, "dice": 0.0, "dce": 0.0, "edge": 0.0}

        self.scaler.scale(loss / float(self._accum_n)).backward()
        self._accum_step += 1
        if self._accum_step >= self._accum_n:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._accum_step = 0
            self.ema.update(self.model)
        return loss, loss_dict

    def _forward_loss(self, images, labels, dwm_t):
        mask_logits, edge_logits, deep_mask = self.model(images)
        target_size = labels.shape[1:]
        if mask_logits.shape[2:] != target_size:
            mask_logits = F.interpolate(mask_logits, size=target_size, mode="trilinear", align_corners=False)
        if edge_logits.shape[2:] != target_size:
            edge_logits = F.interpolate(edge_logits, size=target_size, mode="trilinear", align_corners=False)
        deep_mask = [
            F.interpolate(dm, size=target_size, mode="trilinear", align_corners=False)
            if dm.shape[2:] != target_size else dm for dm in deep_mask
        ]
        return self.criterion(mask_logits, edge_logits, deep_mask, labels, dwm_t)

    @staticmethod
    def _sliding_starts(size, window, stride):
        if size <= window:
            return [0]
        starts = list(range(0, size - window + 1, stride))
        if starts[-1] != size - window:
            starts.append(size - window)
        return starts

    @torch.no_grad()
    def _gaussian_window(self, patch_size, sigma_scale=0.125):
        pd, ph, pw = patch_size
        sigmas = [max(1.0, sigma_scale * s) for s in (pd, ph, pw)]
        zs = torch.arange(pd, device=self.device, dtype=torch.float32) - (pd - 1) / 2.0
        ys = torch.arange(ph, device=self.device, dtype=torch.float32) - (ph - 1) / 2.0
        xs = torch.arange(pw, device=self.device, dtype=torch.float32) - (pw - 1) / 2.0
        gz = torch.exp(-(zs ** 2) / (2.0 * sigmas[0] ** 2))
        gy = torch.exp(-(ys ** 2) / (2.0 * sigmas[1] ** 2))
        gx = torch.exp(-(xs ** 2) / (2.0 * sigmas[2] ** 2))
        g = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
        return torch.clamp(g, min=g.max() * 1e-3)

    @torch.no_grad()
    def _sliding_window_logits(self, image):
        _, _, d, h, w = image.shape
        pd, ph, pw = self.config.patch_size
        sd, sh, sw = max(1, pd // 2), max(1, ph // 2), max(1, pw // 2)
        logits_sum = torch.zeros((1, self.config.num_classes, d, h, w), dtype=torch.float32, device=self.device)
        weight_sum = torch.zeros((1, 1, d, h, w), dtype=torch.float32, device=self.device)
        gauss = self._gaussian_window((pd, ph, pw))[None, None]
        for z in self._sliding_starts(d, pd, sd):
            for y in self._sliding_starts(h, ph, sh):
                for x in self._sliding_starts(w, pw, sw):
                    patch = image[:, :, z:z + pd, y:y + ph, x:x + pw]
                    with autocast(enabled=self.config.use_amp, dtype=torch.bfloat16):
                        patch_logits = self.model(patch)[0]
                    if patch_logits.shape[2:] != patch.shape[2:]:
                        patch_logits = F.interpolate(patch_logits, size=patch.shape[2:], mode="trilinear", align_corners=False)
                    logits_sum[:, :, z:z + pd, y:y + ph, x:x + pw] += patch_logits.float() * gauss
                    weight_sum[:, :, z:z + pd, y:y + ph, x:x + pw] += gauss
        return logits_sum / torch.clamp(weight_sum, min=1e-8)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        pd, ph, pw = self.config.patch_size
        preds, targets = [], []
        with EMASwap(self.ema, self.model):
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                for b in range(images.size(0)):
                    image = images[b:b + 1]
                    label = labels[b:b + 1]
                    _, _, d, h, w = image.shape
                    pad = (0, max(0, pw - w), 0, max(0, ph - h), 0, max(0, pd - d))
                    if any(pad):
                        image = F.pad(image, pad)
                        label = F.pad(label, pad)
                    pred = torch.argmax(self._sliding_window_logits(image), dim=1)
                    preds.append(pred[:, :d, :h, :w].cpu().numpy())
                    targets.append(label[:, :d, :h, :w].cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        preds = postprocess_segmentation(preds, bone_classes=(1, 3), cartilage_classes=(2, 4, 5))
        return macro_dice(preds.flatten(), targets.flatten(), self.config.num_classes)

    def _log_val(self, epoch, macro):
        path = os.path.join(self.config.checkpoint_dir, "val_metrics.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps({"epoch": epoch + 1, "macro_DSC": macro}) + "\n")

    def train(self, train_loader, val_loader):
        for epoch in range(self.start_epoch, self.config.epochs):
            self.model.train()
            epoch_losses = []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
            for batch in pbar:
                _, loss_dict = self._train_step(batch)
                epoch_losses.append(loss_dict["total"])
                pbar.set_postfix({k: f"{loss_dict.get(k, 0):.4f}" for k in ("total", "dice", "dce", "edge")})

            if self._accum_step > 0:
                try:
                    self.scaler.unscale_(self.optimizer)
                except Exception:
                    pass
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.ema.update(self.model)
                self._accum_step = 0

            self.criterion.step_alpha()
            self.scheduler.step()
            print(f"Epoch {epoch + 1} avg_loss={np.mean(epoch_losses):.4f} "
                  f"lr={self.optimizer.param_groups[0]['lr']:.2e} alpha={self.criterion.alpha:.3f}")

            if (epoch + 1) % self.config.val_interval == 0:
                macro = self.validate(val_loader)
                self._log_val(epoch, macro)
                print(f"Validation macro_DSC={macro:.4f}")
                if macro > self.best_macro:
                    self.best_macro = macro
                    self.best_epoch = epoch + 1
                    self.epochs_without_improvement = 0
                    with EMASwap(self.ema, self.model):
                        self._save(epoch, "best_checkpoint.pth", is_best=True)
                    print(f"New best macro_DSC={macro:.4f}")
                else:
                    self.epochs_without_improvement += self.config.val_interval
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    self._save(epoch, "checkpoint_latest.pth")
                    break

            self._save(epoch, "checkpoint_latest.pth")

        print(f"Training complete. Best macro_DSC={self.best_macro:.4f} at epoch {self.best_epoch}")


def run_experiment(config):
    from .data import build_loaders
    enable_gpu_optimizations()
    train_loader, val_loader = build_loaders(config)
    trainer = MbEdTrainer(config)
    trainer.train(train_loader, val_loader)
    return trainer
