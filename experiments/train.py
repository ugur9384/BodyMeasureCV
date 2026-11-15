"""Fast BMnet training: all data + augmentation on GPU, bf16 autocast.

Usage: python train.py --name exp --iters 145000 [--meta_head] [--prior] [--gender] [--scale_aug 0.15] ...
Outputs in OUT_DIR/<name>/: best_model.pth, preds.npz, metrics.json, log.txt
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

MEAS = ["ankle", "arm-length", "bicep", "calf", "chest", "forearm", "height", "hip",
        "leg-length", "shoulder-breadth", "shoulder-to-crotch", "thigh", "waist", "wrist"]
KEY_IDX = [MEAS.index(m) for m in ("chest", "waist", "hip")]

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--data", default="/content/data/bodym_packed.npz")
p.add_argument("--out", default="/content/runs")
p.add_argument("--iters", type=int, default=145000)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=0.0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--full", action="store_true", help="train on train+val, keep last checkpoint")
p.add_argument("--backbone", default="mnasnet", choices=["mnasnet", "mnasnet13", "effb0", "effb2", "regnety16"])
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--meta_head", action="store_true", help="also feed h/w/bmi(/gender) into the MLP head")
p.add_argument("--gender", action="store_true", help="use gender as input")
p.add_argument("--prior", action="store_true", help="predict residual over linear h/w/gender prior")
p.add_argument("--no_meta_ch", action="store_true", help="drop constant h/w image channels")
p.add_argument("--aug", type=int, default=1)
p.add_argument("--rot", type=float, default=3.0)
p.add_argument("--trans", type=float, default=20.0)
p.add_argument("--scale_aug", type=float, default=0.0, help="random zoom +-x (needs meta to recover scale)")
p.add_argument("--morph_p", type=float, default=0.5)
p.add_argument("--hw_noise", type=float, default=0.0, help="gaussian noise on height(cm)/weight(kg) inputs")
p.add_argument("--loss", default="l1", choices=["l1", "smoothl1"])
p.add_argument("--key_w", type=float, default=1.0, help="loss weight for chest/waist/hip")
p.add_argument("--val_every", type=int, default=2000)
p.add_argument("--compile", type=int, default=1)
p.add_argument("--eval_bs", type=int, default=64)
p.add_argument("--precise_bn", type=int, default=1)
p.add_argument("--synth", default="", help="npz from gen_synth.py; trained with a separate head")
p.add_argument("--synth_frac", type=float, default=0.5, help="fraction of each batch from synthetic data")
p.add_argument("--synth_w", type=float, default=1.0)
p.add_argument("--bn_mom", type=float, default=0.01, help="torchvision MNASNet default is 0.0003 (stale running stats)")
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
out_dir = os.path.join(args.out, args.name)
os.makedirs(out_dir, exist_ok=True)
logf = open(os.path.join(out_dir, "log.txt"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


log("ARGS", json.dumps(vars(args)))

# ---------------- data ----------------
D = np.load(args.data)
BITS = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=dev)


def to_gpu(prefix, idx=None):
    sl = slice(None) if idx is None else idx
    return dict(
        front=torch.from_numpy(D[f"{prefix}_front"][sl]).to(dev),
        lat=torch.from_numpy(D[f"{prefix}_lat"][sl]).to(dev),
        y=torch.from_numpy(D[f"{prefix}_y"][sl]).to(dev),
        hw=torch.from_numpy(D[f"{prefix}_hw"][sl]).to(dev),
        male=torch.from_numpy(D[f"{prefix}_male"][sl]).to(dev),
    )


is_val = D["train_is_val"]
excl = D["train_excluded"]
tr_idx = np.where(~excl)[0] if args.full else np.where(~is_val & ~excl)[0]
va_idx = np.where(is_val)[0]
TR = to_gpu("train", tr_idx)
VA = to_gpu("train", va_idx)
TA = to_gpu("testA")
TB = to_gpu("testB")
log(f"train {len(tr_idx)} val {len(va_idx)} testA {len(TA['y'])} testB {len(TB['y'])}")
SYN_COLS = [0, 1, 3, 2]  # chest, waist_slice, hip, waist_loop
if args.synth:
    _S = np.load(args.synth)
    _ok = np.isfinite(_S["meas"][:, SYN_COLS]).all(1)
    SY = dict(front=torch.from_numpy(_S["front"][_ok]).to(dev), lat=torch.from_numpy(_S["lat"][_ok]).to(dev),
              y=torch.from_numpy(_S["meas"][_ok][:, SYN_COLS]).to(dev),
              hw=torch.from_numpy(_S["hw"][_ok]).float().to(dev), male=torch.from_numpy(_S["male"][_ok]).float().to(dev))
    log(f"synth {len(SY['y'])} samples, target means {SY['y'].mean(0).tolist()}")


def unpack(packed):  # (B,640,60) uint8 -> (B,640,480) float
    return ((packed.unsqueeze(-1) & BITS) > 0).flatten(-2).float()


# ---------------- prior ----------------
def prior_feats(hw, male):
    h, w = hw[:, 0] / 100.0, hw[:, 1] / 100.0
    bmi = w / (h * h) / 100.0
    cols = [torch.ones_like(h), h, w, bmi, h * w, w * w, h * h]
    if args.gender:
        cols += [male, male * h, male * w]
    return torch.stack(cols, 1)


if args.prior:
    X = prior_feats(TR["hw"], TR["male"]).double()
    Y = (TR["y"] / 100.0).double()
    PRIOR_W = torch.linalg.solve(X.T @ X + 1e-6 * torch.eye(X.shape[1], device=dev, dtype=torch.float64), X.T @ Y)
    RES_STD = (Y - X @ PRIOR_W).std(0).float()  # per-measurement residual std (m)
    if args.synth:
        Xs = prior_feats(SY["hw"], SY["male"]).double()
        Ys = (SY["y"] / 100.0).double()
        PRIOR_WS = torch.linalg.solve(Xs.T @ Xs + 1e-6 * torch.eye(Xs.shape[1], device=dev, dtype=torch.float64), Xs.T @ Ys)
        RES_STD_S = (Ys - Xs @ PRIOR_WS).std(0).float()
    log("residual std (mm)", (RES_STD * 1000).round().tolist())
else:
    PRIOR_W = None
    PRIOR_WS = None
    RES_STD = RES_STD_S = 1.0


def prior_of(hw, male, W=None):
    W = PRIOR_W if W is None else W
    if W is None:
        return 0.0
    # float64: features are collinear, TF32 matmul would destroy the fit
    return (prior_feats(hw.double(), male.double()) @ W).float()


# ---------------- augmentation ----------------
def augment(x):  # x: (B,1,H,W) binary float
    B = x.shape[0]
    flip = (torch.rand(B, device=dev) < 0.5).float() * -2 + 1
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(args.rot)
    sc = 1 + (torch.rand(B, device=dev) * 2 - 1) * args.scale_aug
    H, W = x.shape[-2:]
    tx = (torch.rand(B, device=dev) * 2 - 1) * args.trans * 2 / W
    ty = (torch.rand(B, device=dev) * 2 - 1) * args.trans * 2 / H
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    # rotation in pixel space with anisotropic normalized coords
    theta = torch.stack([
        torch.stack([cos * flip, -sin * H / W, tx], 1),
        torch.stack([sin * W / H * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    if args.morph_p > 0:
        m = torch.rand(B, device=dev)
        dil = F.max_pool2d(x, 3, 1, 1)
        ero = -F.max_pool2d(-x, 3, 1, 1)
        sel_d = (m < args.morph_p / 2).view(B, 1, 1, 1)
        sel_e = ((m >= args.morph_p / 2) & (m < args.morph_p)).view(B, 1, 1, 1)
        x = torch.where(sel_d, dil, torch.where(sel_e, ero, x))
    return x


def build_input(S, idx, train):
    f = unpack(S["front"][idx]).unsqueeze(1)
    l = unpack(S["lat"][idx]).unsqueeze(1)
    if train and args.aug:
        f, l = augment(f), augment(l)
    sil = torch.cat([f, l], dim=3)  # (B,1,640,960)
    hw = S["hw"][idx].clone()
    if train and args.hw_noise > 0:
        hw = hw + torch.randn_like(hw) * args.hw_noise
    male = S["male"][idx]
    chans = [sil]
    if not args.no_meta_ch:
        chans += [(hw[:, 0] / 200.0).view(-1, 1, 1, 1).expand_as(sil),
                  (hw[:, 1] / 150.0).view(-1, 1, 1, 1).expand_as(sil)]
        if args.gender:
            chans.append(male.view(-1, 1, 1, 1).expand_as(sil))
    else:
        chans += [sil, sil]
    meta = torch.stack([hw[:, 0] / 200.0, hw[:, 1] / 150.0,
                        hw[:, 1] / (hw[:, 0] / 100.0) ** 2 / 40.0] + ([male] if args.gender else []), 1)
    return torch.cat(chans, 1).contiguous(memory_format=torch.channels_last), meta, hw, male


# ---------------- model ----------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        in_ch = 3 + (1 if (args.gender and not args.no_meta_ch) else 0)
        if args.backbone.startswith("mnasnet"):
            base = (models.mnasnet1_0(weights=models.MNASNet1_0_Weights.DEFAULT) if args.backbone == "mnasnet"
                    else models.mnasnet1_3(weights=models.MNASNet1_3_Weights.DEFAULT))
            feats = base.layers
            nf = 1280
        elif args.backbone.startswith("eff"):
            base = (models.efficientnet_b0(weights="DEFAULT") if args.backbone == "effb0"
                    else models.efficientnet_b2(weights="DEFAULT"))
            feats = base.features
            nf = base.classifier[1].in_features
        else:
            base = models.regnet_y_1_6gf(weights="DEFAULT")
            feats = nn.Sequential(base.stem, base.trunk_output)
            nf = 888
        if in_ch != 3:  # widen the first conv; extra channels start at zero
            parent, attr, old = next((p, n, c) for p in feats.modules() for n, c in p.named_children()
                                     if isinstance(c, nn.Conv2d))
            new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, :3] = old.weight
            setattr(parent, attr, new)
        self.features = feats
        self.pool = nn.AdaptiveAvgPool2d(1)
        nm = (4 if args.gender else 3) if args.meta_head else 0
        self.head = nn.Sequential(nn.Linear(nf + nm, args.hidden), nn.ReLU(inplace=True),
                                  nn.Linear(args.hidden, 14))
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                nn.init.zeros_(m.bias)
        if args.synth:
            self.head_s = nn.Sequential(nn.Linear(nf + nm, args.hidden), nn.ReLU(inplace=True),
                                        nn.Linear(args.hidden, len(SYN_COLS)))
        if args.prior:
            nn.init.zeros_(self.head[2].weight)
            if args.synth:
                nn.init.zeros_(self.head_s[2].weight)

    def embed(self, x, meta):
        z = self.pool(self.features(x)).flatten(1)
        if args.meta_head:
            z = torch.cat([z, meta], 1)
        return z

    def forward(self, x, meta):
        return self.head(self.embed(x, meta))


model = Net().to(dev).to(memory_format=torch.channels_last)
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = args.bn_mom
fwd_embed = torch.compile(model.embed) if args.compile else model.embed


def fwd(x, meta):
    return model.head(fwd_embed(x, meta))
opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
sched = torch.optim.lr_scheduler.MultiStepLR(
    opt, [int(args.iters * 0.75), int(args.iters * 0.88)], gamma=0.1)
lw = torch.ones(14, device=dev)
lw[KEY_IDX] = args.key_w
lw = lw / lw.mean()


@torch.no_grad()
def precise_bn(n_batches=60):
    """Recompute BN running stats on un-augmented train batches with current weights."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average
    model.train()
    g = torch.Generator(device=dev).manual_seed(0)
    for _ in range(n_batches):
        idx = torch.randint(0, n, (args.bs,), device=dev, generator=g)
        x, meta, _, _ = build_input(TR, idx, False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, meta)
    for m in bns:
        m.momentum = args.bn_mom


@torch.no_grad()
def predict(S, bs=None):
    bs = bs or args.eval_bs
    model.eval()
    outs = []
    n = len(S["y"])
    for i in range(0, n, bs):
        idx = torch.arange(i, min(n, i + bs), device=dev)
        x, meta, hw, male = build_input(S, idx, False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = fwd(x, meta).float()
        outs.append((o * RES_STD + prior_of(hw, male)) * 100.0)
    model.train()
    return torch.cat(outs)  # cm


def metrics(pred, y):
    err = pred - y
    ae = err.abs() * 10.0
    r = {"MAE": ae.mean().item(),
         "TP50": torch.quantile(ae, 0.5, dim=0).mean().item(),
         "TP75": torch.quantile(ae, 0.75, dim=0).mean().item(),
         "TP90": torch.quantile(ae, 0.9, dim=0).mean().item()}
    for m, j in zip(("chest", "waist", "hip"), KEY_IDX):
        r[f"{m}_MAE"] = ae[:, j].mean().item()
        r[f"{m}_bias"] = err[:, j].mean().item() * 10
    e3 = err[:, KEY_IDX]
    r["pass"] = ((e3 >= -1) & (e3 <= 2)).all(1).float().mean().item() * 100
    return {k: round(v, 3) for k, v in r.items()}


# ---------------- train ----------------
n = len(tr_idx)
best, best_it = float("inf"), 0
t0 = time.time()
run_loss = 0.0
start_it = 1
last_path = os.path.join(out_dir, "last.pth")
if os.path.exists(last_path):  # resume after a lost VM
    ck = torch.load(last_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    start_it, best, best_it = ck["it"] + 1, ck["best"], ck["best_it"]
    log(f"RESUME from it {ck['it']} best {best:.3f}@{best_it}")
for it in range(start_it, args.iters + 1):
    nr = args.bs - (int(round(args.bs * args.synth_frac)) if args.synth else 0)
    idx = torch.randint(0, n, (nr,), device=dev)
    x, meta, hw, male = build_input(TR, idx, True)
    target = (TR["y"][idx] / 100.0 - (prior_of(TR["hw"][idx], male) if args.prior else 0.0)) / RES_STD
    if args.synth:
        sidx = torch.randint(0, len(SY["y"]), (args.bs - nr,), device=dev)
        xs, metas, _, males = build_input(SY, sidx, True)
        target_s = (SY["y"][sidx] / 100.0 - (prior_of(SY["hw"][sidx], males, PRIOR_WS) if args.prior else 0.0)) / RES_STD_S
        x, meta = torch.cat([x, xs]), torch.cat([meta, metas])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = fwd_embed(x, meta)
        out = model.head(z[:nr]).float()
        out_s = model.head_s(z[nr:]).float() if args.synth else None
    if args.loss == "l1":
        l = (out - target).abs()
    else:
        l = F.smooth_l1_loss(out, target, beta=0.005, reduction="none")
    loss = (l * lw).mean()
    if args.synth:
        loss = loss + args.synth_w * (out_s - target_s).abs().mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    run_loss += loss.item() if it % 100 == 0 else 0.0
    if it % 1000 == 0:
        el = time.time() - t0
        log(f"it {it} loss {loss.item():.5f} lr {opt.param_groups[0]['lr']:.1e} "
            f"{it / el:.1f} it/s eta {(args.iters - it) / (it / el) / 3600:.2f}h")
    if (not args.full and it % args.val_every == 0) or it == args.iters:
        if not args.full:
            saved = {k: v.clone() for k, v in model.state_dict().items() if "running" in k}
            if args.precise_bn:
                precise_bn()
            mv = metrics(predict(VA), VA["y"])
            if mv["MAE"] < best:
                best, best_it = mv["MAE"], it
                torch.save({"model_state_dict": model.state_dict(), "iteration": it, "args": vars(args),
                            "prior_w": PRIOR_W.cpu() if PRIOR_W is not None else None,
                            "res_std": RES_STD.cpu() if torch.is_tensor(RES_STD) else None,
                            "prior_ws": PRIOR_WS.cpu() if (args.synth and PRIOR_WS is not None) else None,
                            "best_val_mae": best}, os.path.join(out_dir, "best_model.pth"))
            log(f"VAL it {it} {json.dumps(mv)} best {best:.3f}@{best_it}")
            model.load_state_dict(saved, strict=False)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "it": it,
                    "best": best, "best_it": best_it}, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)

if args.full:
    if args.precise_bn:
        precise_bn()
    torch.save({"model_state_dict": model.state_dict(), "iteration": args.iters, "args": vars(args),
                "prior_w": PRIOR_W.cpu() if PRIOR_W is not None else None,
                "res_std": RES_STD.cpu() if torch.is_tensor(RES_STD) else None},
               os.path.join(out_dir, "best_model.pth"))
else:
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pth"))["model_state_dict"])

res, preds = {}, {}
for nm, S in [("val", VA), ("testA", TA), ("testB", TB)]:
    pr = predict(S)
    preds[nm] = pr.cpu().numpy()
    res[nm] = metrics(pr, S["y"])
    if nm != "val":
        cm = torch.from_numpy(D[f"{nm}_clean_mask"]).to(dev)
        res[nm + "_clean"] = metrics(pr[cm], S["y"][cm])
for k, v in res.items():
    log("FINAL", k, json.dumps(v))
np.savez(os.path.join(out_dir, "preds.npz"), **preds)
json.dump({"args": vars(args), "best_it": best_it, "results": res},
          open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
log("DONE")
