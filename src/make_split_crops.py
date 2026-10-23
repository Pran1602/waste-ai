import yaml, glob, os
from PIL import Image

ROOT, SIZE, MAXN = "data/raw/ds4", 64, 600
names = yaml.safe_load(open(f"{ROOT}/data.yaml"))["names"]
if isinstance(names, dict):
    names = [names[i] for i in sorted(names)]

for split, out in (("valid", "val"), ("test", "test")):
    count = {n: 0 for n in names}
    for lf in glob.glob(f"{ROOT}/{split}/labels/*.txt"):
        stem = os.path.splitext(os.path.basename(lf))[0]
        ip = next((f"{ROOT}/{split}/images/{stem}{e}" for e in (".jpg", ".jpeg", ".png")
                   if os.path.exists(f"{ROOT}/{split}/images/{stem}{e}")), None)
        if not ip:
            continue
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        for line in open(lf):
            p = line.split()
            if len(p) != 5:
                continue
            n = names[int(p[0])]
            if count[n] >= MAXN:
                continue
            cx, cy, w, h = [float(x) for x in p[1:]]
            s = max(w * W, h * H) * 1.1
            if s < 48:
                continue
            box = (int(cx * W - s / 2), int(cy * H - s / 2), int(cx * W + s / 2), int(cy * H + s / 2))
            d = f"data/cls/{out}/{n}"
            os.makedirs(d, exist_ok=True)
            im.crop(box).resize((SIZE, SIZE)).save(f"{d}/{n}_{count[n]}.jpg")
            count[n] += 1
    print(split, count)