# ============================================================
# Server-ready: CIFAR-10 cGAN Ablation
# - Headless-safe (no plt.show)
# - Save ALL figures/images as .jpg
# - Outputs: ./outputs_cgan_ablation_jpg
# ============================================================

import os, math, random, time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image

# ---- Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional (FID/KID). In Colab, this usually works.
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    TORCHMETRICS_OK = True
except Exception as e:
    TORCHMETRICS_OK = False
    print("[warn] torchmetrics FID/KID not available:", e)

# -----------------------------
# Reproducibility
# -----------------------------
def seed_all(seed=17):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(17)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# -----------------------------
# Output directory (jpg only)
# -----------------------------
OUTDIR = "./outputs_cgan_ablation_jpg"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainCfg:
    run_name: str
    disc_type: str          # "concat" | "proj"
    loss_type: str          # "bce" | "hinge_sn"
    z_dim: int = 128
    emb_dim: int = 128
    batch_size: int = 256
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: tuple = (0.5, 0.999)
    steps: int = 30000
    d_steps: int = 1
    img_size: int = 32
    num_classes: int = 10
    save_every: int = 2000

# -----------------------------
# Data: CIFAR-10
# -----------------------------
transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),  # -> [-1,1]
])

# NOTE: If server has no internet, set download=False and pre-place data in ./data
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
test_loader  = DataLoader(testset,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

# -----------------------------
# Utils
# -----------------------------
def denorm(x):
    # [-1,1] -> [0,1]
    return (x.clamp(-1,1) + 1) / 2

def save_grid_jpg(imgs, nrow=10, path="grid.jpg", dpi=200):
    """
    Save a 10x10 grid as .jpg.
    We use matplotlib to force jpg output reliably.
    """
    grid = make_grid(denorm(imgs), nrow=nrow)  # (C,H,W), float [0,1]
    grid_np = grid.permute(1,2,0).detach().cpu().numpy()
    plt.figure(figsize=(10,10))
    plt.imshow(grid_np)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, format="jpg")
    plt.close()

def save_loss_curve_jpg(results, key, title, path, dpi=220):
    plt.figure(figsize=(10,4))
    for name, r in results.items():
        plt.plot(r["losses"][key], label=f"{name}:{key}", alpha=0.85)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, format="jpg")
    plt.close()

def save_cm_jpg(cm, title, path, normalize=True, dpi=250):
    cm = cm.astype(np.float32)
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Target")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, format="jpg")
    plt.close()

# -----------------------------
# Models: DCGAN-ish Generator (fixed across all runs)
# Conditioning in G via concat(z, embed(y))
# -----------------------------
class GNet(nn.Module):
    def __init__(self, z_dim=128, num_classes=10, emb_dim=128, ch=64):
        super().__init__()
        self.embed = nn.Embedding(num_classes, emb_dim)

        in_dim = z_dim + emb_dim
        self.fc = nn.Linear(in_dim, 4*4*ch*8)

        self.net = nn.Sequential(
            nn.BatchNorm2d(ch*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ch*8, ch*4, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ch*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ch*4, ch*2, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ch*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ch*2, ch, 4, 2, 1, bias=False),    # 32x32
            nn.BatchNorm2d(ch),
            nn.ReLU(True),

            nn.Conv2d(ch, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, y):
        yemb = self.embed(y)
        x = torch.cat([z, yemb], dim=1)
        x = self.fc(x).view(x.size(0), -1, 4, 4)
        return self.net(x)

# -----------------------------
# Discriminator Variant 1: Concat D
# -----------------------------
class DConcat(nn.Module):
    def __init__(self, num_classes=10, emb_dim=128, ch=64, use_sn=False):
        super().__init__()
        self.embed = nn.Embedding(num_classes, emb_dim)

        def C(in_c, out_c, k=4, s=2, p=1):
            conv = nn.Conv2d(in_c, out_c, k, s, p)
            return spectral_norm(conv) if use_sn else conv

        self.conv = nn.Sequential(
            C(3 + emb_dim, ch,   4,2,1),
            nn.LeakyReLU(0.2, True),

            C(ch,   ch*2, 4,2,1),
            nn.LeakyReLU(0.2, True),

            C(ch*2, ch*4, 4,2,1),
            nn.LeakyReLU(0.2, True),

            C(ch*4, ch*8, 4,2,1),
            nn.LeakyReLU(0.2, True),
        )
        lin = nn.Linear(ch*8*2*2, 1)
        self.fc = spectral_norm(lin) if use_sn else lin

    def forward(self, x, y):
        yemb = self.embed(y).unsqueeze(-1).unsqueeze(-1)
        ymap = yemb.expand(-1, -1, x.size(2), x.size(3))
        h = self.conv(torch.cat([x, ymap], dim=1))
        h = h.view(h.size(0), -1)
        return self.fc(h).squeeze(1)

# -----------------------------
# Discriminator Variant 2: Projection D
# -----------------------------
class DProj(nn.Module):
    def __init__(self, num_classes=10, emb_dim=128, ch=64, use_sn=False):
        super().__init__()

        def C(in_c, out_c, k=4, s=2, p=1):
            conv = nn.Conv2d(in_c, out_c, k, s, p)
            return spectral_norm(conv) if use_sn else conv

        self.conv = nn.Sequential(
            C(3,    ch,   4,2,1),
            nn.LeakyReLU(0.2, True),

            C(ch,   ch*2, 4,2,1),
            nn.LeakyReLU(0.2, True),

            C(ch*2, ch*4, 4,2,1),
            nn.LeakyReLU(0.2, True),

            C(ch*4, ch*8, 4,2,1),
            nn.LeakyReLU(0.2, True),
        )
        feat_dim = ch*8*2*2
        lin_f = nn.Linear(feat_dim, 1)
        lin_h = nn.Linear(feat_dim, emb_dim)
        self.f = spectral_norm(lin_f) if use_sn else lin_f
        self.h = spectral_norm(lin_h) if use_sn else lin_h

        emb = nn.Embedding(num_classes, emb_dim)
        self.embed = spectral_norm(emb) if use_sn else emb

    def forward(self, x, y):
        hmap = self.conv(x).view(x.size(0), -1)
        out_f = self.f(hmap).squeeze(1)
        out_h = self.h(hmap)
        yemb  = self.embed(y)
        proj = (out_h * yemb).sum(dim=1)
        return out_f + proj

# -----------------------------
# Losses
# -----------------------------
def d_loss_bce(d_real, d_fake):
    return F.softplus(-d_real).mean() + F.softplus(d_fake).mean()

def g_loss_bce(d_fake):
    return F.softplus(-d_fake).mean()

def d_loss_hinge(d_real, d_fake):
    return F.relu(1. - d_real).mean() + F.relu(1. + d_fake).mean()

def g_loss_hinge(d_fake):
    return (-d_fake).mean()

# -----------------------------
# Quick CIFAR-10 Classifier for label-consistency metric
# -----------------------------
def train_quick_classifier(epochs=3, lr=1e-3):
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    return model

print("Training quick classifier for label-consistency metric...")
clf = train_quick_classifier(epochs=3, lr=1e-3)
print("Classifier ready.")

@torch.no_grad()
def label_consistency_acc(generator, n_per_class=200, z_dim=128):
    generator.eval()
    total, correct = 0, 0
    for y in range(10):
        yb = torch.full((n_per_class,), y, device=device, dtype=torch.long)
        z = torch.randn(n_per_class, z_dim, device=device)
        xg = generator(z, yb)
        pred = clf(xg).argmax(dim=1)  # clf trained on normalized [-1,1]
        correct += (pred == yb).sum().item()
        total += n_per_class
    return correct / total

@torch.no_grad()
def diversity_proxy(generator, n_per_class=200, z_dim=128):
    generator.eval()
    feats, labels = [], []
    hook_feats = {}

    def hook_fn(m, inp, out):
        hook_feats["x"] = out

    h = clf.avgpool.register_forward_hook(hook_fn)

    for y in range(10):
        yb = torch.full((n_per_class,), y, device=device, dtype=torch.long)
        z = torch.randn(n_per_class, z_dim, device=device)
        xg = generator(z, yb)
        _ = clf(xg)
        f = hook_feats["x"].flatten(1)  # (B, 512)
        feats.append(f.detach().cpu())
        labels.append(torch.full((n_per_class,), y))

    h.remove()
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    var_sum = 0.0
    for y in range(10):
        fy = feats[labels == y]
        var_sum += fy.var(dim=0, unbiased=False).mean().item()
    return var_sum / 10.0

# -----------------------------
# FID/KID (Optional)
# -----------------------------
@torch.no_grad()
def compute_fid_kid(generator, n_gen=2000, z_dim=128):
    if not TORCHMETRICS_OK:
        return None, None

    generator.eval()
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size=1000, normalize=True).to(device)

    # Real
    n_real = 0
    for xb, _ in test_loader:
        xb = xb.to(device)
        fid.update(denorm(xb), real=True)
        kid.update(denorm(xb), real=True)
        n_real += xb.size(0)
        if n_real >= n_gen:
            break

    # Fake
    gen_bs, n_done = 256, 0
    while n_done < n_gen:
        b = min(gen_bs, n_gen - n_done)
        z = torch.randn(b, z_dim, device=device)
        y = torch.randint(0, 10, (b,), device=device)
        xg = generator(z, y)
        fid.update(denorm(xg), real=False)
        kid.update(denorm(xg), real=False)
        n_done += b

    fid_val = float(fid.compute().item())
    kid_mean, kid_std = kid.compute()
    return fid_val, (float(kid_mean.item()), float(kid_std.item()))

# -----------------------------
# Training
# -----------------------------
def build_models(cfg: TrainCfg):
    G = GNet(z_dim=cfg.z_dim, num_classes=cfg.num_classes, emb_dim=cfg.emb_dim).to(device)

    use_sn = (cfg.loss_type == "hinge_sn")
    if cfg.disc_type == "concat":
        D = DConcat(num_classes=cfg.num_classes, emb_dim=cfg.emb_dim, use_sn=use_sn).to(device)
    elif cfg.disc_type == "proj":
        D = DProj(num_classes=cfg.num_classes, emb_dim=cfg.emb_dim, use_sn=use_sn).to(device)
    else:
        raise ValueError("disc_type must be concat|proj")

    return G, D

def train_one(cfg: TrainCfg):
    G, D = build_models(cfg)

    optG = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=cfg.betas)
    optD = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=cfg.betas)

    losses = {"d": [], "g": []}
    fixed_z = torch.randn(100, cfg.z_dim, device=device)
    fixed_y = torch.tensor([i for i in range(10) for _ in range(10)], device=device, dtype=torch.long)

    loader_iter = iter(train_loader)

    for step in range(1, cfg.steps + 1):
        try:
            real, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            real, y = next(loader_iter)

        real, y = real.to(device), y.to(device)
        b = real.size(0)

        # ---- D
        for _ in range(cfg.d_steps):
            z = torch.randn(b, cfg.z_dim, device=device)
            fake = G(z, y).detach()

            d_real = D(real, y)
            d_fake = D(fake, y)
            dloss = d_loss_bce(d_real, d_fake) if cfg.loss_type == "bce" else d_loss_hinge(d_real, d_fake)

            optD.zero_grad(set_to_none=True)
            dloss.backward()
            optD.step()

        # ---- G
        z = torch.randn(b, cfg.z_dim, device=device)
        fake = G(z, y)
        d_fake2 = D(fake, y)
        gloss = g_loss_bce(d_fake2) if cfg.loss_type == "bce" else g_loss_hinge(d_fake2)

        optG.zero_grad(set_to_none=True)
        gloss.backward()
        optG.step()

        losses["d"].append(float(dloss.item()))
        losses["g"].append(float(gloss.item()))

        if step % cfg.save_every == 0 or step == 1:
            with torch.no_grad():
                G.eval()
                sample = G(fixed_z, fixed_y).detach().cpu()
                G.train()

            # Save intermediate samples as jpg
            sample_path = os.path.join(OUTDIR, f"samples_{cfg.run_name}_step{step:06d}.jpg")
            save_grid_jpg(sample, nrow=10, path=sample_path, dpi=200)
            print(f"[{cfg.run_name}] step {step}/{cfg.steps} | D {dloss.item():.3f} | G {gloss.item():.3f} | saved {sample_path}")

    with torch.no_grad():
        G.eval()
        sample = G(fixed_z, fixed_y).detach().cpu()
    return G, D, losses, sample

# -----------------------------
# Confusion matrix helper (saved as jpg)
# -----------------------------
@torch.no_grad()
def gen_confusion(generator, n_per_class=200, z_dim=128):
    cm = torch.zeros(10, 10, dtype=torch.int64)
    generator.eval()
    for y in range(10):
        yb = torch.full((n_per_class,), y, device=device, dtype=torch.long)
        z = torch.randn(n_per_class, z_dim, device=device)
        xg = generator(z, yb)
        pred = clf(xg).argmax(dim=1)
        for p in pred.detach().cpu().tolist():
            cm[y, p] += 1
    return cm.cpu().numpy()

# -----------------------------
# Run Ablation Grid
# -----------------------------
runs = [
    TrainCfg(run_name="concat_bce",     disc_type="concat", loss_type="bce"),
    TrainCfg(run_name="concat_hingeSN", disc_type="concat", loss_type="hinge_sn"),
    TrainCfg(run_name="proj_bce",       disc_type="proj",   loss_type="bce"),
    TrainCfg(run_name="proj_hingeSN",   disc_type="proj",   loss_type="hinge_sn"),
]

results = {}

for cfg in runs:
    print("\n=== RUN:", cfg.run_name, "===\n")
    G, D, losses, sample = train_one(cfg)

    # Metrics
    lc = label_consistency_acc(G, n_per_class=100, z_dim=cfg.z_dim)
    div = diversity_proxy(G, n_per_class=100, z_dim=cfg.z_dim)
    fid_val, kid_val = compute_fid_kid(G, n_gen=2000, z_dim=cfg.z_dim)

    # Save final sample grid as jpg
    final_grid_path = os.path.join(OUTDIR, f"samples_{cfg.run_name}_final.jpg")
    save_grid_jpg(sample, nrow=10, path=final_grid_path, dpi=220)

    # Save confusion matrix as jpg
    cm = gen_confusion(G, n_per_class=100, z_dim=cfg.z_dim)
    cm_path = os.path.join(OUTDIR, f"cm_{cfg.run_name}.jpg")
    save_cm_jpg(cm, title=f"{cfg.run_name} confusion (normalized)", path=cm_path, normalize=True, dpi=250)

    results[cfg.run_name] = {
        "cfg": cfg,
        "G": G,
        "D": D,
        "losses": losses,
        "sample": sample,
        "label_consistency": lc,
        "diversity_proxy": div,
        "fid": fid_val,
        "kid": kid_val,
        "final_grid_path": final_grid_path,
        "cm_path": cm_path,
    }

    print(f"[saved] {final_grid_path}")
    print(f"[saved] {cm_path}")

# -----------------------------
# Save loss curves as jpg
# -----------------------------
lossD_path = os.path.join(OUTDIR, "loss_D.jpg")
lossG_path = os.path.join(OUTDIR, "loss_G.jpg")
save_loss_curve_jpg(results, "d", "Discriminator loss curves", lossD_path)
save_loss_curve_jpg(results, "g", "Generator loss curves", lossG_path)
print("[saved]", lossD_path)
print("[saved]", lossG_path)

# -----------------------------
# Metrics printout
# -----------------------------
def fmt_kid(kid):
    if kid is None:
        return "N/A"
    m, s = kid
    return f"{m:.4f} ± {s:.4f}"

print("\n===== METRICS (higher label-consistency is better; lower FID/KID is better) =====")
for name, r in results.items():
    fid_str = "N/A" if r["fid"] is None else f"{r['fid']:.2f}"
    print(f"{name:14s} | "
          f"LC-Acc: {r['label_consistency']:.3f} | "
          f"DivProxy: {r['diversity_proxy']:.4f} | "
          f"FID: {fid_str:>6s} | "
          f"KID: {fmt_kid(r['kid'])}")

print("\nDone. Outputs saved in:", OUTDIR)