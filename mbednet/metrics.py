import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def postprocess_segmentation(volume_pred, bone_classes=(1, 3), cartilage_classes=(2, 4, 5)):
    arr = np.asarray(volume_pred)
    squeeze = (arr.ndim == 3)
    if squeeze:
        arr = arr[None]
    out = arr.copy()
    for b in range(arr.shape[0]):
        for c in bone_classes:
            mask = (out[b] == c)
            if not mask.any():
                continue
            labeled, n = ndimage.label(mask)
            if n > 1:
                sizes = ndimage.sum(mask, labeled, range(1, n + 1))
                largest = int(np.argmax(sizes)) + 1
                out[b][mask & (labeled != largest)] = 0
        for c in cartilage_classes:
            mask = (out[b] == c)
            if not mask.any():
                continue
            closed = ndimage.binary_closing(mask, iterations=1)
            new_voxels = closed & ~mask & (out[b] == 0)
            out[b][new_voxels] = c
    return out[0] if squeeze else out


def largest_cc_all(pred, classes=(1, 2, 3, 4, 5)):
    out = pred.copy()
    for c in classes:
        mask = (out == c)
        if not mask.any():
            continue
        labeled, n = ndimage.label(mask)
        if n > 1:
            sizes = ndimage.sum(mask, labeled, range(1, n + 1))
            largest = int(np.argmax(sizes)) + 1
            out[mask & (labeled != largest)] = 0
    return out


def dice_score(pred, target, class_id):
    pc = (pred == class_id)
    tc = (target == class_id)
    denom = int(pc.sum()) + int(tc.sum())
    return float(2.0 * int((pc & tc).sum()) / denom) if denom > 0 else float("nan")


def macro_dice(pred_flat, target_flat, num_classes):
    dices = []
    for c in range(1, num_classes):
        d = dice_score(pred_flat, target_flat, c)
        if d == d:
            dices.append(d)
    return float(np.mean(dices)) if dices else 0.0


class SurfaceMetrics:
    def __init__(self, voxel_spacing=(0.7, 0.365, 0.365)):
        self.voxel_spacing = np.array(voxel_spacing)

    @staticmethod
    def _surface(binary_mask):
        eroded = ndimage.binary_erosion(binary_mask, iterations=1)
        surface = binary_mask.astype(bool) & ~eroded.astype(bool)
        return np.argwhere(surface).astype(np.float64)

    def distances(self, pred_3d, target_3d, class_id):
        pred_bin = (pred_3d == class_id).astype(np.uint8)
        tgt_bin = (target_3d == class_id).astype(np.uint8)
        if pred_bin.sum() == 0 or tgt_bin.sum() == 0:
            return {"HD_mm": float("nan"), "HD95_mm": float("nan"), "ASSD_mm": float("nan")}
        pred_pts = self._surface(pred_bin) * self.voxel_spacing
        tgt_pts = self._surface(tgt_bin) * self.voxel_spacing
        if len(pred_pts) == 0 or len(tgt_pts) == 0:
            return {"HD_mm": float("nan"), "HD95_mm": float("nan"), "ASSD_mm": float("nan")}
        d_p2t, _ = cKDTree(tgt_pts).query(pred_pts, k=1)
        d_t2p, _ = cKDTree(pred_pts).query(tgt_pts, k=1)
        all_d = np.concatenate([d_p2t, d_t2p])
        return {
            "HD_mm": float(all_d.max()),
            "HD95_mm": float(np.percentile(all_d, 95)),
            "ASSD_mm": float(0.5 * (d_p2t.mean() + d_t2p.mean())),
        }
