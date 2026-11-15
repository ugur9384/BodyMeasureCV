"""Evaluate BMnet checkpoints on BodyM TestA/TestB (paper metrics + user pass-rate).

Pass criterion (per user): for chest, waist and hip, error = pred - gt must be in [-1, +2] cm.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MEAS = ["ankle", "arm-length", "bicep", "calf", "chest", "forearm", "height", "hip",
        "leg-length", "shoulder-breadth", "shoulder-to-crotch", "thigh", "waist", "wrist"]
KEY = ["chest", "waist", "hip"]
LO, HI = -1.0, 2.0  # cm
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BMnet(nn.Module):
    def __init__(self, in_ch=3, backbone="mnasnet", laplace=False):
        super().__init__()
        if backbone == "mnasnet":
            base = models.mnasnet1_0(weights=None)
            self.features = base.layers
            if in_ch != 3:
                self.features[0] = nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False)
            nf = 1280
        else:
            base = models.efficientnet_b2(weights=None)
            self.features = base.features
            nf = 1408
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.laplace = laplace
        if laplace:
            self.hidden = nn.Sequential(nn.Linear(nf, 128), nn.ReLU(inplace=True))
            self.output = nn.Linear(128, 28)
        else:
            self.head = nn.Sequential(nn.Linear(nf, 128), nn.ReLU(inplace=True), nn.Linear(128, 14))

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        if self.laplace:
            return self.output(self.hidden(x))[:, :14]
        return self.head(x)


def build_from_ckpt(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    laplace = "output.weight" in sd
    backbone = "effnet" if "features.0.0.weight" in sd else "mnasnet"
    in_ch = sd["features.0.weight"].shape[1] if backbone == "mnasnet" else 3
    m = BMnet(in_ch, backbone, laplace)
    m.load_state_dict(sd)
    return m.to(DEV).eval(), in_ch


def load_mask(path):
    img = np.array(Image.open(path).convert("L"))
    img = np.where(img > 127, 255, 0).astype(np.uint8)
    return cv2.resize(img, (480, 640), interpolation=cv2.INTER_NEAREST)


class TestSet(Dataset):
    def __init__(self, d):
        d = Path(d)
        df = pd.read_csv(d / "subject_to_photo_map.csv").merge(
            pd.read_csv(d / "hwg_metadata.csv"), on="subject_id").merge(
            pd.read_csv(d / "measurements.csv"), on="subject_id")
        ok = [(d / "mask" / f"{p}.png").exists() and (d / "mask_left" / f"{p}.png").exists()
              for p in df.photo_id]
        self.df = df[ok].reset_index(drop=True)
        self.d = d

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        f = load_mask(self.d / "mask" / f"{r.photo_id}.png")
        l = load_mask(self.d / "mask_left" / f"{r.photo_id}.png")
        sil = np.concatenate([f, l], axis=1)
        male = 1.0 if str(r.gender).lower() == "male" else 0.0
        return torch.from_numpy(sil), torch.tensor([r.height_cm, r.weight_kg, male], dtype=torch.float32)


def make_input(sil_u8, meta, in_ch, gender_val=None):
    x = sil_u8.float().div(255.0).unsqueeze(1)
    chans = [x, (meta[:, 0] / 200.0).view(-1, 1, 1, 1).expand_as(x),
             (meta[:, 1] / 150.0).view(-1, 1, 1, 1).expand_as(x)]
    if in_ch == 4:
        g = meta[:, 2] if gender_val is None else gender_val(meta[:, 2])
        chans.append(g.view(-1, 1, 1, 1).expand_as(x))
    return torch.cat(chans, 1)


@torch.no_grad()
def predict(model, in_ch, loader, gender_val=None):
    out = []
    for sil, meta in loader:
        inp = make_input(sil, meta, in_ch, gender_val).to(DEV, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(inp).float().cpu())
    return torch.cat(out).numpy() * 100.0  # cm


def report(pred_cm, df):
    gt = df[MEAS].values
    err = pred_cm - gt
    ae = np.abs(err) * 10.0  # mm
    res = {
        "MAE": ae.mean(),
        "TP50": np.quantile(ae, 0.5, axis=0).mean(),
        "TP75": np.quantile(ae, 0.75, axis=0).mean(),
        "TP90": np.quantile(ae, 0.9, axis=0).mean(),
    }
    for m in KEY:
        j = MEAS.index(m)
        res[f"{m}_MAE"] = ae[:, j].mean()
        res[f"{m}_bias"] = err[:, j].mean() * 10
        res[f"{m}_in"] = ((err[:, j] >= LO) & (err[:, j] <= HI)).mean() * 100
    idx = [MEAS.index(m) for m in KEY]
    e3 = err[:, idx]
    res["pass_photo"] = ((e3 >= LO) & (e3 <= HI)).all(1).mean() * 100
    # subject level: average predictions over the subject's photos
    sub = pd.DataFrame(e3, columns=KEY).groupby(df.subject_id.values).mean()
    res["pass_subj"] = ((sub.values >= LO) & (sub.values <= HI)).all(1).mean() * 100
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("BODYM_ROOT", "data/raw"))
    ap.add_argument("--ckpts", nargs="*")
    ap.add_argument("--out", default="eval_results.json")
    a = ap.parse_args()
    root = Path(a.root)
    sets = {
        "A_orig": root / "BodyM_Dev/bodym_noisy/amazon-bodym/testA",
        "B_orig": root / "BodyM_Dev/bodym_noisy/amazon-bodym/testB",
        "A_clean": root / "BodyM_Dev/bodym_dataset/bodym/testA",
        "B_clean": root / "BodyM_Dev/bodym_dataset/bodym/testB",
    }
    ckpts = a.ckpts or sorted(set(glob.glob(str(root / "**/best_model.pth"), recursive=True)))
    loaders = {k: DataLoader(TestSet(v), batch_size=16, num_workers=8) for k, v in sets.items()}
    results = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for c in ckpts:
        name = os.path.relpath(c, root)
        model, in_ch = build_from_ckpt(c)
        results[name] = {}
        for k, ld in loaders.items():
            pred = predict(model, in_ch, ld)
            results[name][k] = {kk: round(float(v), 2) for kk, v in report(pred, ld.dataset.df).items()}
            r = results[name][k]
            print(f"{name:70s} {k:8s} MAE {r['MAE']:5.2f} TP50 {r['TP50']:5.2f} TP90 {r['TP90']:5.2f} | "
                  f"ch {r['chest_MAE']:5.1f}({r['chest_bias']:+.1f}) wa {r['waist_MAE']:5.1f}({r['waist_bias']:+.1f}) "
                  f"hip {r['hip_MAE']:5.1f}({r['hip_bias']:+.1f}) | pass photo {r['pass_photo']:4.1f}% subj {r['pass_subj']:4.1f}%",
                  flush=True)
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
