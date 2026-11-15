# BodyMeasureCV

Predicting 14 body measurements from two silhouette images (front and side) plus height and weight.
This is my reimplementation of **BMnet** from the BodyM paper
([Ruiz et al., *Human Body Measurement Estimation with Adversarial Augmentation*](https://adversarialbodysim.github.io/files/paper.pdf)), trained on Amazon's BodyM dataset.
The paper's code is not public, so the model was rebuilt from the paper's description.

Measurements: ankle, arm length, bicep, calf, chest, forearm, height, hip, leg length,
shoulder breadth, shoulder-to-crotch, thigh, waist, wrist.

## Results

Main model: **Run 2** (MNASNet backbone, L1 loss, on-the-fly augmentation, targets scaled by /100, no z-score).
All errors are in millimeters, averaged over all 14 measurements, on the **full** BodyM test sets (no images removed).

| Model | TestA MAE | TestA TP50 / TP75 / TP90 | TestB MAE |
|---|---|---|---|
| Run 2 (`notebooks/bmnet_run2.ipynb`) | 9.31 | 7.49 / 13.10 / 19.55 | 12.39 |
| `mp_norm` (`experiments/train.py`, late fusion + gender + linear prior, 40k iters) | 9.07 | 7.32 / 18.98 (TP50 / TP90) | 12.01 |
| Paper, Table 2 (multi-view + height/weight) | – | 7.34 / 12.55 / 18.42 | – |
| Paper, Table 3 (multi-view, real data only) | – | 6.51 / 11.06 / 16.45 | – |

TPxx = the xx-th percentile of absolute error.

Notes, stated honestly:
- Run 2 is close to the paper's Table 2 but about 15–19% behind Table 3. Both tables describe the same setup and the paper does not explain the gap, so I do not call this a successful reproduction.
- All numbers are from a **single seed**. Differences under ~0.3 mm between models may be noise.
- Run 2 was chosen as the main model because it is better on chest, waist and hip (mean 14.07 mm vs. 14.86 mm for `mp_norm`), even though `mp_norm` has the lower overall MAE.
- Run 2 differs from the paper in batch size (32 vs. 22) and iterations (145k vs. 150k).

### Findings
- **z-score target normalization + L1 loss hurt.** With z-scored targets, L1 weights each measurement by 1/σ, so the model focuses on low-variance measurements and the error in mm goes up. The model before Run 1 got 9.43 mm on TestA, Run 1 (z-score + augmentation) got 10.23 mm, and Run 2 (augmentation without z-score) got 9.31 mm. This is a single-seed result.
- **Input height is noisy.** The height given as input differs from the height measured in the 3D scan by 14.3 mm on TestA and 21.0 mm on TestB (per photo). This sets an error floor and explains part of the higher TestB error.
- **Sensitivity to height input.** Adding +5 cm to the input height raises Run 2's TestA MAE from 9.31 to 13.01 mm.

## Repository layout

```
notebooks/bmnet_run2.ipynb   Run 2: data loading, augmentation, training and evaluation (Colab)
experiments/prep_data.py     packs BodyM masks + metadata into one .npz for fast training
experiments/train.py         faster training script used for the later experiments (mp_norm etc.)
experiments/eval_all.py      evaluates checkpoints on TestA / TestB
experiments/analyze.py       per-measurement analysis of saved predictions
splits/                      subject-level train/val splits and the excluded training subjects
```

The code in `notebooks/` is mine. The scripts in `experiments/` were written with the help of an AI coding assistant for the later experiments.

## Data

The dataset is **not** included. Download BodyM from the AWS Open Data registry
(<https://registry.opendata.aws/bodym/>, `s3://amazon-bodym`):

```bash
aws s3 cp --no-sign-request --recursive s3://amazon-bodym data/raw/BodyM_Dev/bodym_noisy/amazon-bodym
```

The scripts expect `BODYM_ROOT` (default `data/raw`) to contain:
- `BodyM_Dev/bodym_noisy/amazon-bodym/{train,testA,testB}`: the original download.
- `BodyM_Dev/bodym_dataset/bodym/{train,testA,testB}`: a copy where I removed photos with broken masks by hand. Run 2 was trained on this copy. The filter is manual and not yet written down as a rule, so this copy is not reproducible from this repository yet. The results above are on the original, unfiltered test sets.

## Weights

Run 2 weights are attached to the [v0.1 release](../../releases/tag/v0.1).
They were trained on BodyM, which is licensed as **Creative Commons Attribution-NonCommercial 4.0**,
so the weights may only be used for **non-commercial** purposes.

## License

Code: Apache License 2.0 (see `LICENSE`). The license does not cover the BodyM dataset or the trained weights.
