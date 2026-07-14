import torch
import torch.nn as nn
import torch.nn.functional as F


class MbEdLoss(nn.Module):
    def __init__(self, num_classes=6, alpha_init=0.9, alpha_min=0.55,
                 alpha_decay=0.005, edge_lambda=0.1, deep_supervision_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha_init
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.edge_lambda = edge_lambda
        self.deep_weights = deep_supervision_weights or [0.4, 0.2, 0.1]
        self.register_buffer("dice_weights", torch.tensor([0.01, 3.0, 4.0, 3.0, 6.0, 5.0]))

    def step_alpha(self):
        self.alpha = max(self.alpha_min, self.alpha - self.alpha_decay)

    def _dice_loss(self, pred, target, smooth=1e-6):
        pred_soft = F.softmax(pred, dim=1)
        losses = []
        for c in range(self.num_classes):
            pc = pred_soft[:, c]
            tc = (target == c).float()
            inter = (pc * tc).sum()
            union = pc.sum() + tc.sum()
            losses.append(self.dice_weights[c] * (1.0 - (2 * inter + smooth) / (union + smooth)))
        return torch.stack(losses).sum() / self.dice_weights[1:].sum()

    def _distanced_ce(self, pred, target, dwm_t):
        ce = F.cross_entropy(pred, target, reduction="none")
        return (ce * dwm_t).mean()

    @staticmethod
    def _compute_edge_gt(target):
        max_c = int(target.max().item())
        edge = torch.zeros((target.size(0), 1, *target.shape[1:]),
                           device=target.device, dtype=torch.float32)
        for c in range(1, max_c + 1):
            tc = (target == c).float().unsqueeze(1)
            eroded = -F.max_pool3d(-tc, kernel_size=3, stride=1, padding=1)
            edge_c = (tc - eroded).clamp_(0.0, 1.0)
            edge = torch.maximum(edge, edge_c)
        return edge

    def _edge_loss(self, edge_logits, target, dwm_t):
        edge_gt = self._compute_edge_gt(target)
        dwm_edge = dwm_t.unsqueeze(1) if dwm_t.dim() == 4 else dwm_t
        dwm_edge = dwm_edge.float()
        dwm_edge = dwm_edge / dwm_edge.mean().clamp_min(1e-6)
        dwm_edge = dwm_edge.clamp(0.25, 4.0)
        edge_frac = edge_gt.mean().detach().clamp(1e-4, 0.5)
        class_w = torch.where(edge_gt > 0.5, 1.0 - edge_frac, edge_frac)
        class_w = class_w / class_w.mean().clamp_min(1e-6)
        bce = F.binary_cross_entropy_with_logits(edge_logits, edge_gt, reduction="none")
        return (bce * dwm_edge * class_w).mean()

    def forward(self, mask_logits, edge_logits, deep_mask_list, target, dwm_t):
        l_dice = self._dice_loss(mask_logits, target)
        l_dce = self._distanced_ce(mask_logits, target, dwm_t)
        l_edge = self._edge_loss(edge_logits, target, dwm_t)
        total = self.alpha * l_dice + (1.0 - self.alpha) * l_dce + self.edge_lambda * l_edge

        for i, pred_deep in enumerate(deep_mask_list):
            w = self.deep_weights[i] if i < len(self.deep_weights) else 0.05
            d_dice = self._dice_loss(pred_deep, target)
            d_ce = F.cross_entropy(pred_deep, target)
            total = total + w * (self.alpha * d_dice + (1.0 - self.alpha) * d_ce)

        comp = {
            "total": total.item(),
            "dice": l_dice.item(),
            "dce": l_dce.item(),
            "edge": l_edge.item(),
            "alpha": self.alpha,
        }
        return total, comp
