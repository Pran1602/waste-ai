import os, random, shutil, glob
random.seed(0)
N_REAL = 100                                   # low-data setting: har class ke 100 real crops
classes = sorted(os.listdir("data/gan"))

for tag in ("A", "B"):
    shutil.rmtree(f"data/cls_{tag}", ignore_errors=True)
    for split in ("val", "test"):
        for c in classes:
            dst = f"data/cls_{tag}/{split}/{c}"
            os.makedirs(dst, exist_ok=True)
            for f in glob.glob(f"data/cls/{split}/{c}/*.jpg"):
                shutil.copy(f, dst)

for c in classes:
    files = sorted(glob.glob(f"data/gan/{c}/img/*.jpg"))
    random.shuffle(files)
    for tag in ("A", "B"):
        dst = f"data/cls_{tag}/train/{c}"
        os.makedirs(dst, exist_ok=True)
        for f in files[:N_REAL]:
            shutil.copy(f, dst)
    for f in glob.glob(f"data/gan_synth/{c}/*.png"):      # sirf B mein synthetic
        shutil.copy(f, f"data/cls_B/train/{c}")

for tag in ("A", "B"):
    print(tag, {c: len(os.listdir(f"data/cls_{tag}/train/{c}")) for c in classes})