"""Pack BodyM masks (640x480, bit-packed) + metadata into one npz for fast Colab training."""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(os.environ.get("BODYM_ROOT", "data/raw"))  # see README for the expected layout
ORIG = ROOT / "BodyM_Dev/bodym_noisy/amazon-bodym"
CLEAN = ROOT / "BodyM_Dev/bodym_dataset/bodym"
HERE = Path(__file__).parent
SPLIT = HERE.parent / "splits/experiments_split.json"
EXCLUDE = HERE.parent / "splits/exclude_train.json"
OUT = HERE / "data/bodym_packed.npz"
MEAS = ["ankle", "arm-length", "bicep", "calf", "chest", "forearm", "height", "hip",
        "leg-length", "shoulder-breadth", "shoulder-to-crotch", "thigh", "waist", "wrist"]


def load(p):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(np.where(img > 127, 255, 0).astype(np.uint8), (480, 640),
                     interpolation=cv2.INTER_NEAREST)
    return np.packbits(img > 0, axis=1)  # (640, 60)


def table(d):
    df = pd.read_csv(d / "subject_to_photo_map.csv").merge(
        pd.read_csv(d / "hwg_metadata.csv"), on="subject_id").merge(
        pd.read_csv(d / "measurements.csv"), on="subject_id")
    ok = [(d / "mask" / f"{p}.png").exists() and (d / "mask_left" / f"{p}.png").exists() for p in df.photo_id]
    return df[ok].reset_index(drop=True)


def main():
    OUT.parent.mkdir(exist_ok=True)
    arrays, meta = {}, {}
    for name, d in [("train", ORIG / "train"), ("testA", ORIG / "testA"), ("testB", ORIG / "testB")]:
        df = table(d)
        with ThreadPoolExecutor(12) as ex:
            fr = list(ex.map(load, [d / "mask" / f"{p}.png" for p in df.photo_id]))
            la = list(ex.map(load, [d / "mask_left" / f"{p}.png" for p in df.photo_id]))
        arrays[f"{name}_front"] = np.stack(fr)
        arrays[f"{name}_lat"] = np.stack(la)
        arrays[f"{name}_y"] = df[MEAS].values.astype(np.float32)
        arrays[f"{name}_hw"] = df[["height_cm", "weight_kg"]].values.astype(np.float32)
        arrays[f"{name}_male"] = (df.gender.str.lower() == "male").values.astype(np.float32)
        meta[f"{name}_subject"] = df.subject_id.tolist()
        meta[f"{name}_photo"] = df.photo_id.astype(str).tolist()
        print(name, len(df))
    # cleaned test subsets = photos still present in the cleaned dataset
    for name in ["testA", "testB"]:
        keep = set(table(CLEAN / name).photo_id.astype(str))
        arrays[f"{name}_clean_mask"] = np.array([p in keep for p in meta[f"{name}_photo"]])
        print(name, "clean", arrays[f"{name}_clean_mask"].sum())
    split = json.load(open(SPLIT))
    excl = set(json.load(open(EXCLUDE)))
    val = set(split["val_ids"])
    arrays["train_is_val"] = np.array([s in val for s in meta["train_subject"]])
    arrays["train_excluded"] = np.array([s in excl for s in meta["train_subject"]])
    np.savez_compressed(OUT, **arrays)
    json.dump(meta, open(OUT.with_suffix(".json"), "w"))
    print("saved", OUT, OUT.stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
