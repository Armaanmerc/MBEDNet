# MbEdNet: Mamba-Edge Network for Knee MRI Segmentation

3D segmentation of femoral bone, femoral cartilage, tibial bone, and tibial cartilage
from knee MRI (OAI-ZIB-CM), trained with 5-fold cross-validation.

The network couples a CNN branch (residual blocks with multi-scale pyramid pooling) and a
tri-orientated Mamba branch, fused per stage by a channel-gated block, with a parallel
edge decoder that refines cartilage boundaries. Training uses a Dice + distance-weighted
cross-entropy + edge loss with deep supervision.

## Setup

```
pip install -r requirements.txt
```

Requires a CUDA GPU (mamba-ssm). Expected dataset layout:

```
OAI-ZIB-CM/
  imagesTr/*_0000.nii.gz   labelsTr/*.nii.gz
  imagesTs/*_0000.nii.gz   labelsTs/*.nii.gz
  info/subInfo_train.csv   info/subInfo_test.csv
```

Optional `.npy` sidecars (`<case>_0000.npy`, `<case>.npy`) next to the NIfTI files are
loaded into a shared RAM cache to avoid repeated network-filesystem reads.

## Train

```
python train.py --fold 0 --data-path /path/to/OAI-ZIB-CM
```

Run folds 0-4 to complete the cross-validation. Each run writes `best_checkpoint.pth`
(EMA weights) and `checkpoint_latest.pth` (restart-safe) to `checkpoints/fold{n}/`.

## Evaluate

```
python evaluate.py --checkpoint checkpoints/fold0/best_checkpoint.pth --fold 0 \
    --data-path /path/to/OAI-ZIB-CM
```

Reports per-structure and macro DSC, HD, HD95, and ASSD on the held-out test fold, for
both the 5-class and 4-class (merged tibial cartilage) settings, with largest-connected-
component post-processing.

## Configuration

Split: pooled 507 scans, KL-stratified `StratifiedKFold(5, seed=35)`; each fold is the
held-out test set with an inner validation split for early stopping.
Patch `64x192x192` at `0.7x0.365x0.365` mm, AdamW `1e-4` with cosine decay, bfloat16 AMP,
EMA `0.99`, effective batch 8.
