# utils.py
import os
import torch
import torchvision.utils as vutils


def one_hot(labels, num_classes=10, device=None):
    if device is None:
        device = labels.device
    y = torch.zeros(labels.size(0), num_classes, device=device)
    y.scatter_(1, labels.view(-1, 1), 1.0)
    return y


@torch.no_grad()
def save_sample_grid(G, device, epoch, z_dim=100, n_classes=10, n_per_class=8, outdir="results/samples"):
    """
    Saves a grid: 10 rows (classes) x n_per_class columns.
    """
    os.makedirs(outdir, exist_ok=True)

    # Create labels: 0..9 repeated n_per_class times
    labels = torch.arange(n_classes).repeat_interleave(n_per_class).to(device)
    y = one_hot(labels, n_classes, device=device)

    # Noise per image
    z = torch.randn(labels.size(0), z_dim, device=device)

    fake = G(z, y)  # (N,1,28,28) in [-1,1]
    # Make a grid with n_per_class columns
    grid = vutils.make_grid(fake, nrow=n_per_class, normalize=True, value_range=(-1, 1))
    path = os.path.join(outdir, f"epoch_{epoch:03d}.png")
    vutils.save_image(grid, path)


def save_checkpoint(path, epoch, G, D, optG, optD):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "optG_state": optG.state_dict(),
        "optD_state": optD.state_dict(),
    }, path)


def append_log_csv(path, row_dict, header_order):
    """
    Appends a row to CSV, creating file with header if needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(",".join(header_order) + "\n")
        f.write(",".join(str(row_dict[h]) for h in header_order) + "\n")
