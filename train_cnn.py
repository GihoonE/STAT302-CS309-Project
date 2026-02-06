# train_cnn.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset, Dataset
from torchvision import datasets, transforms

from cnn_model import SimpleMNISTCNN


class LabelToTensor(Dataset):
    """
    Wrap a dataset so that labels are always torch.LongTensor.
    This fixes ConcatDataset collate errors (MNIST labels are ints).
    """
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        # MNIST returns int labels; convert to tensor
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)
        else:
            y = y.to(dtype=torch.long)
        return x, y


def load_real_mnist(root="data"):
    tfm = transforms.ToTensor()  # keep [0,1]
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    # Wrap to ensure labels are tensors
    train_ds = LabelToTensor(train_ds)
    test_ds = LabelToTensor(test_ds)
    return train_ds, test_ds


def load_synth(path_images, path_labels):
    imgs = torch.load(path_images)  # [-1,1], tensor [N,1,28,28]
    labels = torch.load(path_labels)

    # Ensure labels tensor long
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        labels = labels.to(dtype=torch.long)

    # Convert images to [0,1]
    imgs = (imgs + 1.0) / 2.0
    imgs = imgs.clamp(0.0, 1.0)

    return TensorDataset(imgs, labels)


def subset_dataset(ds, fraction, seed=42):
    n = len(ds)
    n_sub = int(n * fraction)
    g = torch.Generator().manual_seed(seed)
    sub, _ = random_split(ds, [n_sub, n - n_sub], generator=g)
    return sub


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def train_one_setting(setting_name, train_ds, test_loader, device, args):
    n_total = len(train_ds)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_part, val_part = random_split(train_ds, [n_train, n_val], generator=g)

    # macOS stable
    train_loader = DataLoader(train_part, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_part, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SimpleMNISTCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best_val = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = ce(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 2 == 0 or epoch == args.epochs:
            print(f"[{setting_name}] Epoch {epoch:02d}/{args.epochs}  val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    return best_val, test_acc


def append_result_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("setting,n_real,n_synth,best_val,test_acc\n")
        f.write(",".join(map(str, row)) + "\n")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    real_train, real_test = load_real_mnist(root=args.data_root)
    test_loader = DataLoader(real_test, batch_size=256, shuffle=False, num_workers=0)

    synth_ds = None
    if args.synth_fraction > 0:
        synth_ds = load_synth(args.synth_images, args.synth_labels)
        print("Synthetic loaded:", len(synth_ds))

    out_csv = "results/cnn/summary.csv"

    # A) real only
    real_sub = subset_dataset(real_train, args.real_fraction, seed=args.seed)
    nameA = f"real_{int(args.real_fraction*100)}pct"
    best_val, test_acc = train_one_setting(nameA, real_sub, test_loader, device, args)
    print(f"✅ {nameA}: test_acc={test_acc:.4f}")
    append_result_csv(out_csv, (nameA, len(real_sub), 0, round(best_val, 6), round(test_acc, 6)))

    # B) real + synth
    if synth_ds is not None and args.synth_fraction > 0:
        n_real = len(real_sub)
        n_synth = int(n_real * args.synth_fraction)

        g = torch.Generator().manual_seed(args.seed)
        synth_part, _ = random_split(synth_ds, [n_synth, len(synth_ds) - n_synth], generator=g)

        mix_ds = ConcatDataset([real_sub, synth_part])

        nameB = f"real_{int(args.real_fraction*100)}pct_plus_synth_{int(args.synth_fraction*100)}pct"
        best_val2, test_acc2 = train_one_setting(nameB, mix_ds, test_loader, device, args)
        print(f"✅ {nameB}: test_acc={test_acc2:.4f}")
        append_result_csv(out_csv, (nameB, len(real_sub), n_synth, round(best_val2, 6), round(test_acc2, 6)))

    print("Saved:", out_csv)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--real_fraction", type=float, default=1.0)

    p.add_argument("--synth_images", type=str, default="results/synthetic/syn_images.pt")
    p.add_argument("--synth_labels", type=str, default="results/synthetic/syn_labels.pt")
    p.add_argument("--synth_fraction", type=float, default=0.2)

    args = p.parse_args()
    main(args)
