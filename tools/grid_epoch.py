import os
import math
from PIL import Image, ImageDraw

# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # tools/.. = project root
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
OUT_DIR = os.path.join(PROJECT_ROOT, "images", "grids")
os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = [
    ("baseline", "baseline"),
    ("eta=0.1", "label_noise_p10"),
    ("eta=0.2", "label_noise_p20"),
    ("eta=0.3", "label_noise_p30"),
]

EPOCHS = ["epoch_005", "epoch_010", "epoch_015", "epoch_020", "epoch_025", "epoch_030"]
IMG_NAME = "img_0000.png"  # 공정 비교용 (동일 index)

# 데이터셋별로 "대표 클래스 2개"를 자동 선택
# - animals: cat, lion (없으면 앞에서 2개)
# - mnist: 0, 8 (없으면 앞에서 2개)
# - sports_ball: baseball, basketball (없으면 앞에서 2개)
PREFERRED_CLASSES = {
    "animals": ["cat", "lion"],
    "mnist": ["0", "8"],
    "sports_ball": ["baseball", "basketball"],
}

# -----------------------------
# Helpers
# -----------------------------
def safe_listdir(path):
    if not os.path.isdir(path):
        return []
    return sorted([d for d in os.listdir(path) if not d.startswith(".")])

def load_img(path, size=None):
    im = Image.open(path).convert("RGB")
    if size is not None:
        im = im.resize(size)
    return im

def draw_text(draw, xy, text):
    # 폰트 지정 없이도 동작. (필요하면 ImageFont 추가 가능)
    draw.text(xy, text, fill="black")

def make_grid_image(matrix, row_labels, col_labels, cell_size, pad=6, top_pad=40, left_pad=120, bg=(255,255,255)):
    """
    matrix: rows x cols of PIL Images
    """
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    w, h = cell_size

    W = left_pad + cols*w + (cols-1)*pad
    H = top_pad + rows*h + (rows-1)*pad

    canvas = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(canvas)

    # column labels
    for j, lab in enumerate(col_labels):
        x = left_pad + j*(w+pad) + 5
        y = 10
        draw_text(draw, (x, y), str(lab))

    # rows + paste
    for i, rlab in enumerate(row_labels):
        y0 = top_pad + i*(h+pad)
        draw_text(draw, (10, y0 + h//2 - 8), str(rlab))
        for j in range(cols):
            x0 = left_pad + j*(w+pad)
            canvas.paste(matrix[i][j], (x0, y0))

    return canvas

def pick_two_classes(dataset_key, class_list):
    prefs = PREFERRED_CLASSES.get(dataset_key, [])
    chosen = []
    for p in prefs:
        if p in class_list:
            chosen.append(p)
    # 부족하면 앞에서 채움
    for c in class_list:
        if len(chosen) >= 2:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen[:2]

# -----------------------------
# Main per-dataset generation
# -----------------------------
def make_noise_compare_grids(dataset_key, epochs_to_make=("epoch_010","epoch_020","epoch_030")):
    """
    여러 epoch에 대해 noise 비교 grid 생성
    x: baseline/eta
    y: classes (최대 10개)
    """
    outs = []

    # classes는 baseline의 마지막 epoch 기준으로 가져옴 (없으면 첫 epoch로 fallback)
    probe_epoch = epochs_to_make[-1]
    base_epoch_dir = os.path.join(RESULTS_ROOT, dataset_key, "fake_samples", "baseline", probe_epoch)
    classes = safe_listdir(base_epoch_dir)
    if not classes:
        print(f"[SKIP] {dataset_key}: no classes found at {base_epoch_dir}")
        return outs
    classes = classes[:10]

    # cell size probe
    sample_path = os.path.join(RESULTS_ROOT, dataset_key, "fake_samples", "baseline", probe_epoch, classes[0], IMG_NAME)
    if not os.path.exists(sample_path):
        print(f"[SKIP] {dataset_key}: sample not found {sample_path}")
        return outs
    sample = load_img(sample_path)
    cell_size = (sample.size[0], sample.size[1])

    for ep in epochs_to_make:
        matrix = []
        for cls in classes:
            row = []
            for _, cfg_folder in CONFIGS:
                p = os.path.join(RESULTS_ROOT, dataset_key, "fake_samples", cfg_folder, ep, cls, IMG_NAME)
                if not os.path.exists(p):
                    p = os.path.join(RESULTS_ROOT, dataset_key, "fake_samples", "baseline", ep, cls, IMG_NAME)
                if not os.path.exists(p):
                    p = os.path.join(RESULTS_ROOT, dataset_key, "fake_samples", "baseline", probe_epoch, cls, IMG_NAME)
                row.append(load_img(p, size=cell_size))
            matrix.append(row)

        col_labels = [lab for lab, _ in CONFIGS]
        row_labels = classes
        grid = make_grid_image(matrix, row_labels, col_labels, cell_size)
        out = os.path.join(OUT_DIR, f"{dataset_key}_noise_compare_{ep}.png")
        grid.save(out)
        print("[OK]", out)
        outs.append(out)

    return outs


def make_epoch_progression_grids(dataset_key):
    """
    클래스 2개 선택해서 각각 grid 생성
    x: epochs
    y: baseline/η
    """
    base_epoch_dir = os.path.join(RESULTS_ROOT, dataset_key, "fake_samples", "baseline", "epoch_030")
    classes = safe_listdir(base_epoch_dir)
    if not classes:
        print(f"[SKIP] {dataset_key}: no classes found at {base_epoch_dir}")
        return []

    chosen_classes = pick_two_classes(dataset_key, classes)

    outs = []
    for cls in chosen_classes:
        # cell size: epoch_030 baseline로 잡음
        sample_path = os.path.join(RESULTS_ROOT, dataset_key, "fake_samples", "baseline", "epoch_030", cls, IMG_NAME)
        if not os.path.exists(sample_path):
            print(f"[SKIP] {dataset_key}/{cls}: sample not found {sample_path}")
            continue
        sample = load_img(sample_path)
        cell_size = (sample.size[0], sample.size[1])

        matrix = []
        row_labels = []
        for cfg_label, cfg_folder in CONFIGS:
            row_labels.append(cfg_label)
            row = []
            for ep in EPOCHS:
                p = os.path.join(RESULTS_ROOT, dataset_key, "fake_samples", cfg_folder, ep, cls, IMG_NAME)
                if not os.path.exists(p):
                    # 누락시 가장 가까운 대체: baseline 같은 epoch, 없으면 epoch_030 baseline
                    p2 = os.path.join(RESULTS_ROOT, dataset_key, "fake_samples", "baseline", ep, cls, IMG_NAME)
                    p3 = os.path.join(RESULTS_ROOT, dataset_key, "fake_samples", "baseline", "epoch_030", cls, IMG_NAME)
                    p = p2 if os.path.exists(p2) else p3
                row.append(load_img(p, size=cell_size))
            matrix.append(row)

        col_labels = [ep.replace("epoch_", "e") for ep in EPOCHS]
        grid = make_grid_image(matrix, row_labels, col_labels, cell_size)
        out = os.path.join(OUT_DIR, f"{dataset_key}_epoch_progress_{cls}.png")
        grid.save(out)
        print("[OK]", out)
        outs.append(out)

    return outs

def main():
    for dataset_key in ["animals", "mnist", "sports_ball"]:
        print("\n=== Dataset:", dataset_key, "===")
        make_noise_compare_grids(dataset_key, epochs_to_make=("epoch_010","epoch_020","epoch_030"))
        make_epoch_progression_grids(dataset_key)     # 2장 (class 2개)

if __name__ == "__main__":
    main()
