# train_cgan.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models import Generator, Discriminator
from utils import one_hot, save_sample_grid, save_checkpoint, append_log_csv
from mnist_mirror import MNISTMirror


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    # Reproducibility
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # MNIST transform: [0,1] → [-1,1]
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # MNIST with HTTPS mirror (NO SSL ISSUES)
    ds = MNISTMirror(
        root="data",
        train=True,
        download=True,
        transform=tfm
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    # Models
    G = Generator(z_dim=args.z_dim, n_classes=10).to(device)
    D = Discriminator(n_classes=10).to(device)

    # Loss + optimizers
    criterion = nn.BCELoss()
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999))


    # Output folders
    os.makedirs("results/samples", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    log_path = "results/logs/losses.csv"
    header = ["epoch", "step", "loss_D", "loss_G", "D_real", "D_fake"]

    # Save initial samples
    save_sample_grid(G, device, epoch=0, z_dim=args.z_dim)

    step = 0
    for epoch in range(1, args.epochs + 1):
        for x_real, y in dl:
            x_real = x_real.to(device)
            y = y.to(device)
            y_oh = one_hot(y, 10, device=device)

            bs = x_real.size(0)

            # ======================
            # Train Discriminator
            # ======================
            z = torch.randn(bs, args.z_dim, device=device)
            x_fake = G(z, y_oh).detach()

            D_real = D(x_real, y_oh)
            D_fake = D(x_fake, y_oh)

            loss_D = (
                criterion(D_real, torch.ones(bs, 1, device=device)) +
                criterion(D_fake, torch.zeros(bs, 1, device=device))
            ) / 2

            optD.zero_grad(set_to_none=True)
            loss_D.backward()
            optD.step()

            # ======================
            # Train Generator
            # ======================
            z = torch.randn(bs, args.z_dim, device=device)
            x_fake = G(z, y_oh)
            D_fake = D(x_fake, y_oh)

            loss_G = criterion(D_fake, torch.ones(bs, 1, device=device))

            optG.zero_grad(set_to_none=True)
            loss_G.backward()
            optG.step()

            step += 1

            if step % args.log_every == 0:
                append_log_csv(
                    log_path,
                    {
                        "epoch": epoch,
                        "step": step,
                        "loss_D": loss_D.item(),
                        "loss_G": loss_G.item(),
                        "D_real": D_real.mean().item(),
                        "D_fake": D_fake.mean().item(),
                    },
                    header
                )
                print(
                    f"[Epoch {epoch:03d} | Step {step:06d}] "
                    f"lossD={loss_D:.4f} lossG={loss_G:.4f} "
                    f"D(real)={D_real.mean():.3f} D(fake)={D_fake.mean():.3f}"
                )

        # Save samples + checkpoint
        save_sample_grid(G, device, epoch, z_dim=args.z_dim)

        if epoch % args.ckpt_every == 0:
            save_checkpoint(
                f"results/checkpoints/cgan_epoch_{epoch:03d}.pt",
                epoch, G, D, optG, optD
            )

    save_checkpoint("results/checkpoints/cgan_final.pt", args.epochs, G, D, optG, optD)
    print("✅ Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--ckpt_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    train(args)
