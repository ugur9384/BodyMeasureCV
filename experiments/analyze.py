"""Analyze downloaded run predictions: ensembles + per-measurement offset calibration for the -1/+2 cm target.

Usage: python analyze.py run1 run2 ...   (expects runs/<name>/preds.npz and metrics.json)
Offsets for chest/waist/hip are fit on the validation split only, then applied to TestA/TestB.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
D = np.load(HERE / "data/bodym_packed.npz")
META = json.load(open(HERE / "data/bodym_packed.json"))
MEAS = ["ankle", "arm-length", "bicep", "calf", "chest", "forearm", "height", "hip",
        "leg-length", "shoulder-breadth", "shoulder-to-crotch", "thigh", "waist", "wrist"]
KEY = [MEAS.index(m) for m in ("chest", "waist", "hip")]
LO, HI = -1.0, 2.0

Y = {"val": D["train_y"][D["train_is_val"]], "testA": D["testA_y"], "testB": D["testB_y"]}
SUBJ = {"val": np.array(META["train_subject"])[D["train_is_val"]],
        "testA": np.array(META["testA_subject"]), "testB": np.array(META["testB_subject"])}


def stats(pred, split, off=np.zeros(3)):
    err = pred - Y[split]
    ae = np.abs(err) * 10
    e3 = err[:, KEY] + off
    inr = (e3 >= LO) & (e3 <= HI)
    return dict(MAE=ae.mean(), TP50=np.quantile(ae, .5, 0).mean(), TP90=np.quantile(ae, .9, 0).mean(),
                ch=ae[:, KEY[0]].mean(), wa=ae[:, KEY[1]].mean(), hip=ae[:, KEY[2]].mean(),
                in_ch=inr[:, 0].mean() * 100, in_wa=inr[:, 1].mean() * 100, in_hip=inr[:, 2].mean() * 100,
                pass_=inr.all(1).mean() * 100)


def fit_offsets(pred):
    """Per-measurement shift maximizing in-window rate on val (coordinate search)."""
    e3 = (pred - Y["val"])[:, KEY]
    grid = np.arange(-3, 3.001, 0.05)
    off = np.zeros(3)
    for _ in range(3):
        for j in range(3):
            best = max(grid, key=lambda o: (((e3 + np.where(np.arange(3) == j, o, off)) >= LO) &
                                           ((e3 + np.where(np.arange(3) == j, o, off)) <= HI)).all(1).mean())
            off[j] = best
    return off


def load(name):
    return dict(np.load(HERE / "runs" / name / "preds.npz"))


def show(tag, P):
    off = fit_offsets(P["val"])
    for sp in ("testA", "testB"):
        r0, r1 = stats(P[sp], sp), stats(P[sp], sp, off)
        print(f"{tag[:40]:40s} {sp}  MAE {r0['MAE']:5.2f} TP50 {r0['TP50']:5.2f} TP90 {r0['TP90']:5.2f} | "
              f"ch {r0['ch']:5.1f} wa {r0['wa']:5.1f} hip {r0['hip']:5.1f} | pass {r0['pass_']:4.1f}% "
              f"-> calib {r1['pass_']:4.1f}% (in ch/wa/hip {r1['in_ch']:.0f}/{r1['in_wa']:.0f}/{r1['in_hip']:.0f}) "
              f"off {np.round(off, 2)}")


if __name__ == "__main__":
    names = sys.argv[1:]
    preds = {n: load(n) for n in names}
    for n, P in preds.items():
        show(n, P)
    if len(names) > 1:
        ens = {k: np.mean([P[k] for P in preds.values()], 0) for k in ("val", "testA", "testB")}
        show("ENSEMBLE(" + ",".join(names) + ")", ens)
