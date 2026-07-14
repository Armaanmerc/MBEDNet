import argparse
import json

import numpy as np
import torch

from mbednet import make_config, MbEdNet, OAIZIBDataset
from mbednet.metrics import SurfaceMetrics, largest_cc_all, dice_score
from mbednet.trainer import MbEdTrainer, enable_gpu_optimizations

NAMES5 = {1: "Femoral_Bone", 2: "Femoral_Cartilage", 3: "Tibial_Bone",
          4: "Medial_Tibial_Cartilage", 5: "Lateral_Tibial_Cartilage"}
NAMES4 = {1: "Femoral_Bone", 2: "Femoral_Cartilage", 3: "Tibial_Bone", 4: "Tibial_Cartilage"}
METRICS = ["DSC", "HD_mm", "HD95_mm", "ASSD_mm"]


def evaluate_fold(checkpoint, fold, data_path):
    cfg = make_config(checkpoint_dir="/tmp/eval", fold=fold, data_path=data_path)
    ds = OAIZIBDataset(cfg, split="test")
    surf = SurfaceMetrics(voxel_spacing=cfg.target_spacing)
    model = MbEdNet(in_channels=cfg.in_channels, num_classes=cfg.num_classes,
                    features=cfg.features).cuda().eval()
    ck = torch.load(checkpoint, map_location="cuda", weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)

    trainer = MbEdTrainer.__new__(MbEdTrainer)
    trainer.config = cfg
    trainer.device = torch.device("cuda")
    trainer.model = model

    acc5 = {c: {m: [] for m in METRICS} for c in range(1, 6)}
    acc4 = {c: {m: [] for m in METRICS} for c in range(1, 5)}
    for i in range(len(ds.samples)):
        item = ds[i]
        img = item["image"].unsqueeze(0).cuda()
        lbl = item["label"].numpy()
        with torch.no_grad():
            pred = torch.argmax(trainer._sliding_window_logits(img), dim=1)[0].cpu().numpy()
        pred = largest_cc_all(pred, classes=(1, 2, 3, 4, 5))
        for c in range(1, 6):
            acc5[c]["DSC"].append(dice_score(pred, lbl, c))
            for m, v in surf.distances(pred, lbl, c).items():
                if v == v:
                    acc5[c][m].append(v)
        pred4 = pred.copy(); pred4[pred4 == 5] = 4
        lbl4 = lbl.copy(); lbl4[lbl4 == 5] = 4
        for c in (1, 2, 3):
            acc4[c]["DSC"].append(dice_score(pred, lbl, c))
            for m, v in surf.distances(pred, lbl, c).items():
                if v == v:
                    acc4[c][m].append(v)
        acc4[4]["DSC"].append(dice_score(pred4, lbl4, 4))
        for m, v in surf.distances(pred4, lbl4, 4).items():
            if v == v:
                acc4[4][m].append(v)

    def summarize(acc, names):
        return {names[c]: {m: (float(np.mean(acc[c][m])) if acc[c][m] else float("nan")) for m in METRICS}
                for c in names}
    return {"four": summarize(acc4, NAMES4), "five": summarize(acc5, NAMES5)}


def show(title, table):
    print(f"\n{title}")
    print(f"  {'structure':26} {'DSC%':>8} {'HD_mm':>8} {'HD95_mm':>8} {'ASSD_mm':>8}")
    macro = {m: [] for m in METRICS}
    for name, vals in table.items():
        print(f"  {name:26} {vals['DSC'] * 100:8.2f} {vals['HD_mm']:8.2f} {vals['HD95_mm']:8.2f} {vals['ASSD_mm']:8.3f}")
        for m in METRICS:
            if vals[m] == vals[m]:
                macro[m].append(vals[m])
    print(f"  {'MACRO':26} {np.mean(macro['DSC']) * 100:8.2f} {np.mean(macro['HD_mm']):8.2f} "
          f"{np.mean(macro['HD95_mm']):8.2f} {np.mean(macro['ASSD_mm']):8.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    enable_gpu_optimizations()
    result = evaluate_fold(args.checkpoint, args.fold, args.data_path)
    show("4-CLASS (tibial cartilage merged)", result["four"])
    show("5-CLASS (medial/lateral separate)", result["five"])
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
