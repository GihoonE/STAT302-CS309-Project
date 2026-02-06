# generate_synthetic.py
import os
import argparse
import torch

from models import Generator
from utils import one_hot


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    G = Generator(z_dim=args.z_dim, n_classes=10).to(device)
    G.load_state_dict(ckpt["G_state"])
    G.eval()

    os.makedirs(args.outdir, exist_ok=True)

    n_classes = 10
    total = args.total
    per_class = total // n_classes
    remainder = total - per_class * n_classes

    all_imgs = []
    all_labels = []

    print(f"Generating {total} images = {per_class} per class (+{remainder} extra distributed)...")

    for c in range(n_classes):
        n_c = per_class + (1 if c < remainder else 0)

        labels = torch.full((n_c,), c, dtype=torch.long, device=device)
        y = one_hot(labels, n_classes, device=device)

        z = torch.randn(n_c, args.z_dim, device=device)
        fake = G(z, y)  # [-1,1], shape (n_c,1,28,28)

        # store on CPU
        all_imgs.append(fake.cpu())
        all_labels.append(labels.cpu())

    imgs = torch.cat(all_imgs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Shuffle once so classes are mixed
    perm = torch.randperm(imgs.size(0))
    imgs = imgs[perm]
    labels = labels[perm]

    # Save
    img_path = os.path.join(args.outdir, "syn_images.pt")
    lbl_path = os.path.join(args.outdir, "syn_labels.pt")
    torch.save(imgs, img_path)
    torch.save(labels, lbl_path)

    print("Saved:")
    print(" ", img_path, imgs.shape, imgs.min().item(), imgs.max().item())
    print(" ", lbl_path, labels.shape)

    # Quick sanity counts
    counts = torch.bincount(labels, minlength=10)
    print("Class counts:", counts.tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="results/checkpoints/cgan_final.pt",
                   help="Path to checkpoint (e.g., results/checkpoints/cgan_final.pt)")
    p.add_argument("--outdir", type=str, default="results/synthetic")
    p.add_argument("--total", type=int, default=12000, help="Total synthetic images to generate")
    p.add_argument("--z_dim", type=int, default=100)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    main(args)
