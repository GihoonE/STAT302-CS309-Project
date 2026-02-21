import os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

OUTDIR = "./outputs_cgan_ablation_jpg"
CIFAR10_NAMES = ["airplane","automobile","bird","cat","deer",
                 "dog","frog","horse","ship","truck"]

def add_labels_to_grid_clean(img_path, out_path, nrow=10, title=None, dpi=300, fontsize=10):
    img = mpimg.imread(img_path)
    H, W = img.shape[:2]
    cell_w = W / nrow
    cell_h = H / nrow

    fig, ax = plt.subplots(figsize=(10,10), dpi=dpi)
    ax.imshow(img)
    ax.set_aspect("equal")

    # ticks at cell centers
    xs = [(j + 0.5) * cell_w for j in range(nrow)]
    ys = [(i + 0.5) * cell_h for i in range(nrow)]

    ax.set_xticks(xs)
    ax.set_xticklabels([str(j) for j in range(nrow)], fontsize=fontsize)
    ax.set_yticks(ys)
    ax.set_yticklabels(CIFAR10_NAMES[:nrow], fontsize=fontsize)

    ax.set_xlabel("Sample index within class", fontsize=fontsize+1)
    ax.set_ylabel("Class", fontsize=fontsize+1)
    if title:
        ax.set_title(title, fontsize=fontsize+2)

    # clean look
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.18, bottom=0.08, right=0.99, top=0.92)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def relabel_all(outdir=OUTDIR, overwrite=False):
    patterns = [
        os.path.join(outdir, "samples_*_final.jpg"),
        os.path.join(outdir, "samples_*_step*.jpg"),
    ]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    paths = sorted(set(paths))

    if not paths:
        print("[warn] no matching grids found in:", outdir)
        return

    print(f"[info] found {len(paths)} grids")

    for in_path in paths:
        base = os.path.basename(in_path)

        # skip already-labeled outputs to avoid infinite loops
        if "_labeled" in base:
            continue

        # output filename
        out_base = base.replace(".jpg", "_labeled_clean.jpg")
        out_path = os.path.join(outdir, out_base)

        if (not overwrite) and os.path.exists(out_path):
            print("[skip exists]", out_base)
            continue

        # nicer title: e.g., samples_proj_bce_step002000.jpg -> proj_bce @ step 2000
        title = None
        if base.startswith("samples_"):
            core = base[len("samples_"):-len(".jpg")]
            if "_step" in core:
                run, step = core.split("_step", 1)
                title = f"{run} @ step {int(step)} (row=class, col=sample)"
            elif core.endswith("_final"):
                run = core[:-len("_final")]
                title = f"{run} final (row=class, col=sample)"
            else:
                title = core

        add_labels_to_grid_clean(in_path, out_path, title=title)
        print("[saved]", out_base)

if __name__ == "__main__":
    relabel_all(OUTDIR, overwrite=False)