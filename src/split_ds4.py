import random, shutil, pathlib

root = pathlib.Path("data/raw/ds4")
imgs = sorted((root / "train/images").glob("*.*"))
random.seed(42)
random.shuffle(imgs)
n = len(imgs)
splits = {"valid": imgs[: int(0.1 * n)],
          "test": imgs[int(0.1 * n): int(0.2 * n)]}

for name, files in splits.items():
    for d in ("images", "labels"):
        (root / name / d).mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.move(str(f), root / name / "images" / f.name)
        lbl = root / "train/labels" / (f.stem + ".txt")
        if lbl.exists():
            shutil.move(str(lbl), root / name / "labels" / lbl.name)
print("total images:", n)