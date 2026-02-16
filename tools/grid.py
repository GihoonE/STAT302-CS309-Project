import os
from PIL import Image

base_root = "results/animals/fake_samples"
configs = {
    "baseline": "baseline",
    "p10": "label_noise_p10",
    "p20": "label_noise_p20",
    "p30": "label_noise_p30"
}

epoch = "epoch_030"

# 클래스 목록 (baseline 기준으로 자동 추출)
class_root = os.path.join(base_root, "baseline", epoch)
classes = sorted(os.listdir(class_root))

img_size = None
images = []

for cls in classes:
    row = []
    for key in configs:
        path = os.path.join(base_root, configs[key], epoch, cls, "img_0000.png")
        img = Image.open(path).convert("RGB")
        if img_size is None:
            img_size = img.size
        row.append(img)
    images.append(row)

# grid 생성
cols = len(configs)
rows = len(classes)
w, h = img_size
pad = 5

grid = Image.new("RGB", (cols*w + (cols-1)*pad, rows*h + (rows-1)*pad), (255,255,255))

for r in range(rows):
    for c in range(cols):
        x = c*(w+pad)
        y = r*(h+pad)
        grid.paste(images[r][c], (x, y))

grid.save("images/animals_noise_comparison_epoch30.png")
print("Saved to images/animals_noise_comparison_epoch30.png")
