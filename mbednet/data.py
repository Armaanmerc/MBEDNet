import csv
import re
import time
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import KL4_SAMPLING_WEIGHTS

try:
    import edt as _edt_pkg

    def _aux_edt(mask_bool):
        return _edt_pkg.edt(np.ascontiguousarray(mask_bool, dtype=np.uint32))
except Exception:
    def _aux_edt(mask_bool):
        return distance_transform_edt(mask_bool)

_VOL_CACHE = {}
DWM_GAMMA = 8.0
DWM_SIGMA_PER_CLASS = {1: 10.0, 2: 3.0, 3: 10.0, 4: 3.0, 5: 3.0}


def cached_npy_load(path_str):
    a = _VOL_CACHE.get(path_str)
    if a is None:
        a = np.load(path_str)
        try:
            a.flags.writeable = False
        except Exception:
            pass
        _VOL_CACHE[path_str] = a
    return a


class OAIZIBDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        self._init_cv_fold(config, split)

    def _load_kl_csv(self, config, csv_name):
        candidates = [
            Path(config.data_path) / "info" / csv_name,
            Path(config.data_path).parent / "info" / csv_name,
        ]
        out = {}
        for p in candidates:
            if not p.exists():
                continue
            with open(p, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                if "CMT-ID" in cols and "KLGrade" in cols:
                    for row in reader:
                        cid = str(row.get("CMT-ID", "")).strip()
                        klr = str(row.get("KLGrade", "")).strip()
                        if cid and klr:
                            try:
                                out[cid] = int(float(klr))
                            except (ValueError, TypeError):
                                pass
            if out:
                break
        return out

    def _init_cv_fold(self, config, split):
        all_samples = []
        for sub_img, sub_lbl in (("imagesTr", "labelsTr"), ("imagesTs", "labelsTs")):
            img_dir = Path(config.data_path) / sub_img
            lbl_dir = Path(config.data_path) / sub_lbl
            for img_file in sorted(img_dir.glob("*_0000.nii.gz")):
                base_name = img_file.name.replace("_0000.nii.gz", "")
                lbl_file = lbl_dir / f"{base_name}.nii.gz"
                if lbl_file.exists():
                    all_samples.append((img_file, lbl_file, base_name))
        if not all_samples:
            raise ValueError(f"No samples found under {config.data_path}")

        self.kl_lookup = {}
        self.kl_lookup.update(self._load_kl_csv(config, "subInfo_train.csv"))
        self.kl_lookup.update(self._load_kl_csv(config, "subInfo_test.csv"))

        def kl_of(s):
            m = re.search(r"(\d+)$", s[2])
            return self.kl_lookup.get(m.group(1), 0) if m else 0

        kl = [kl_of(s) for s in all_samples]
        k = int(config.cv_folds)
        fold = int(config.cv_fold) % k
        seed = int(config.cv_seed)

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        trainval_idx, test_idx = list(skf.split(range(len(all_samples)), kl))[fold]
        tv_kl = [kl[i] for i in trainval_idx]
        try:
            tr_idx, val_idx = train_test_split(
                list(trainval_idx), test_size=config.cv_inner_val_frac,
                stratify=tv_kl, random_state=seed)
        except ValueError:
            tr_idx, val_idx = train_test_split(
                list(trainval_idx), test_size=config.cv_inner_val_frac, random_state=seed)

        chosen = {"train": tr_idx, "val": val_idx, "test": list(test_idx)}.get(split, tr_idx)
        self.samples = [all_samples[i] for i in sorted(chosen)]
        print(f"[{split}] fold {fold + 1}/{k} seed {seed}: {len(self.samples)} samples "
              f"(train {len(tr_idx)} / val {len(val_idx)} / test {len(test_idx)})")

    def __len__(self):
        return len(self.samples) * 2 if self.split == "train" else len(self.samples)

    @staticmethod
    def _compute_dwm(mask_np, num_classes):
        dwm = np.ones_like(mask_np, dtype=np.float32)
        for c in range(1, num_classes):
            tc = (mask_np == c)
            s = int(tc.sum())
            if s == 0 or s == tc.size:
                continue
            edt_min = np.minimum(_aux_edt(tc), _aux_edt(~tc))
            sigma = DWM_SIGMA_PER_CLASS.get(c, 10.0)
            dwm += (DWM_GAMMA * np.exp(-edt_min / sigma)).astype(np.float32)
        return dwm

    def __getitem__(self, idx):
        img_path, lbl_path, base_name = self.samples[idx % len(self.samples)]
        npy_img = img_path.parent / img_path.name.replace("_0000.nii.gz", "_0000.npy")
        npy_lbl = lbl_path.parent / lbl_path.name.replace(".nii.gz", ".npy")
        if npy_img.exists() and npy_lbl.exists():
            mri = cached_npy_load(str(npy_img)).astype(np.float32)
            mask = cached_npy_load(str(npy_lbl)).astype(np.int64)
        else:
            mri = nib.load(img_path).get_fdata().astype(np.float32)
            mask = nib.load(lbl_path).get_fdata().astype(np.float32)

        mri = np.nan_to_num(mri, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64)

        flat = mri.ravel()
        sub = flat[::8] if flat.size > 1_000_000 else flat
        p1, p99 = np.percentile(sub, [1, 99])
        if p99 - p1 > 1e-8:
            mri = np.clip((mri - p1) / (p99 - p1), 0.0, 1.0).astype(np.float32)
        else:
            mri = np.zeros_like(mri, dtype=np.float32)

        input_vol = np.stack([mri], axis=0)
        input_vol = np.nan_to_num(input_vol, nan=0.0, posinf=0.0, neginf=0.0)

        if self.split == "train":
            input_vol, mask = self._random_crop(input_vol, mask)
            input_vol, mask = self._augment(input_vol, mask)
        else:
            input_vol, mask = self._pad_to_min_size(input_vol, mask)

        m = re.search(r"(\d+)$", base_name)
        kl_grade = self.kl_lookup.get(m.group(1), -1) if m else -1
        out = {
            "image": torch.from_numpy(input_vol.copy()),
            "label": torch.from_numpy(mask.copy()).long(),
            "name": base_name,
            "kl_grade": kl_grade,
        }
        if self.split == "train":
            dwm = self._compute_dwm(mask, self.config.num_classes)
            out["dwm"] = torch.from_numpy(dwm)
        return out

    def _random_crop(self, image, mask):
        _, d, h, w = image.shape
        cd, ch, cw = self.config.patch_size
        cartilage_classes = [4, 5, 2]
        cartilage_weights = [0.45, 0.30, 0.25]
        bone_classes = [1, 3]
        bone_weights = [0.5, 0.5]
        min_cart_voxels = 200
        max_attempts = 16

        def try_centered_crop(class_id):
            coords = np.argwhere(mask == class_id)
            if len(coords) == 0:
                return None
            center = coords[np.random.randint(len(coords))]
            jitter = np.array([
                np.random.randint(-cd // 10, cd // 10 + 1),
                np.random.randint(-ch // 10, ch // 10 + 1),
                np.random.randint(-cw // 10, cw // 10 + 1),
            ])
            center = center + jitter
            sd = int(max(0, min(center[0] - cd // 2, d - cd)))
            sh = int(max(0, min(center[1] - ch // 2, h - ch)))
            sw = int(max(0, min(center[2] - cw // 2, w - cw)))
            img_crop = image[:, sd:sd + cd, sh:sh + ch, sw:sw + cw]
            msk_crop = mask[sd:sd + cd, sh:sh + ch, sw:sw + cw]
            if img_crop.shape[1:] != self.config.patch_size:
                return None
            return img_crop, msk_crop

        roll = np.random.random()
        if roll < 0.60:
            for _ in range(max_attempts):
                cls = int(np.random.choice(cartilage_classes, p=cartilage_weights))
                result = try_centered_crop(cls)
                if result is None:
                    continue
                img_crop, msk_crop = result
                cart_vox = int(((msk_crop == 2) | (msk_crop == 4) | (msk_crop == 5)).sum())
                if cart_vox >= min_cart_voxels:
                    return img_crop, msk_crop
        elif roll < 0.85:
            cls = int(np.random.choice(bone_classes, p=bone_weights))
            result = try_centered_crop(cls)
            if result is not None:
                return result

        sd = np.random.randint(0, max(1, d - cd + 1))
        sh = np.random.randint(0, max(1, h - ch + 1))
        sw = np.random.randint(0, max(1, w - cw + 1))
        return (image[:, sd:sd + cd, sh:sh + ch, sw:sw + cw],
                mask[sd:sd + cd, sh:sh + ch, sw:sw + cw])

    def _pad_to_min_size(self, image, mask):
        _, d, h, w = image.shape
        cd, ch, cw = self.config.patch_size
        pad_d, pad_h, pad_w = max(0, cd - d), max(0, ch - h), max(0, cw - w)
        if pad_d or pad_h or pad_w:
            image = np.pad(image, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)))
            mask = np.pad(mask, ((0, pad_d), (0, pad_h), (0, pad_w)))
        return image, mask

    @staticmethod
    def _elastic_deform(image, mask, alpha=200, sigma=20):
        shape = image.shape[1:]
        dx = ndimage.gaussian_filter(np.random.randn(*shape).astype(np.float32), sigma) * alpha
        dy = ndimage.gaussian_filter(np.random.randn(*shape).astype(np.float32), sigma) * alpha
        dz = ndimage.gaussian_filter(np.random.randn(*shape).astype(np.float32), sigma) * alpha
        z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
        coords = [np.clip(z + dz, 0, shape[0] - 1),
                  np.clip(y + dy, 0, shape[1] - 1),
                  np.clip(x + dx, 0, shape[2] - 1)]
        for c in range(image.shape[0]):
            image[c] = ndimage.map_coordinates(image[c], coords, order=1, mode="reflect").astype(np.float32)
        mask = ndimage.map_coordinates(mask, coords, order=0, mode="reflect").astype(np.int64)
        return image, mask

    @staticmethod
    def _augment(image, mask):
        for ax_img, ax_msk in [(1, 0), (2, 1), (3, 2)]:
            if np.random.random() > 0.5:
                image = np.flip(image, axis=ax_img)
                mask = np.flip(mask, axis=ax_msk)
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        if np.random.random() < 0.3:
            image, mask = OAIZIBDataset._elastic_deform(image.copy(), mask.copy())
        if np.random.random() < 0.3:
            angle = np.random.uniform(-15, 15)
            axes = [(1, 2), (1, 3), (2, 3)]
            ax = axes[np.random.randint(len(axes))]
            for c in range(image.shape[0]):
                image[c] = ndimage.rotate(image[c], angle, axes=(ax[0] - 1, ax[1] - 1),
                                          reshape=False, order=1, mode="reflect")
            mask = ndimage.rotate(mask, angle, axes=(ax[0] - 1, ax[1] - 1),
                                  reshape=False, order=0, mode="reflect")
        if np.random.random() < 0.3:
            imin, imax = image.min(), image.max()
            if imax - imin > 1e-8:
                image_01 = (image - imin) / (imax - imin)
                gamma = np.random.uniform(0.7, 1.5)
                image = np.power(image_01, gamma) * (imax - imin) + imin
        if np.random.random() > 0.5:
            image = image * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.1, 0.1)
        if np.random.random() > 0.7:
            image = image + np.random.normal(0, 0.02, image.shape).astype(np.float32)
        return image.astype(np.float32), mask.astype(np.int64)


def build_kl4_sampler(train_ds):
    lookup = getattr(train_ds, "kl_lookup", {}) or {}
    weights = []
    for s in train_ds.samples:
        m = re.search(r"(\d+)$", s[2])
        kl = lookup.get(m.group(1), -1) if m else -1
        weights.append(float(KL4_SAMPLING_WEIGHTS.get(kl, 1.0)))
    return WeightedRandomSampler(np.asarray(weights, dtype=np.float64).tolist(),
                                 num_samples=len(train_ds), replacement=True)


def preload_volumes(datasets):
    paths = set()
    for ds in datasets:
        for img_path, lbl_path, _ in ds.samples:
            i = img_path.parent / img_path.name.replace("_0000.nii.gz", "_0000.npy")
            l = lbl_path.parent / lbl_path.name.replace(".nii.gz", ".npy")
            for p in (i, l):
                if p.exists():
                    paths.add(str(p))
    t0 = time.time()
    n = 0
    for p in sorted(paths):
        if p not in _VOL_CACHE:
            cached_npy_load(p)
            n += 1
    print(f"Preloaded {n} volumes in {time.time() - t0:.0f}s")


def build_loaders(config):
    train_ds = OAIZIBDataset(config, split="train")
    val_ds = OAIZIBDataset(config, split="val")
    try:
        preload_volumes([train_ds, val_ds])
    except Exception as e:
        print(f"Preload skipped: {e}")
    sampler = build_kl4_sampler(train_ds)
    pf = min(2, int(config.prefetch_factor))
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, sampler=sampler,
        num_workers=config.num_workers, pin_memory=True,
        prefetch_factor=pf if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=min(8, config.num_workers), pin_memory=True)
    return train_loader, val_loader
