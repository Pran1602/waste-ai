import os, torch
import torchvision.utils as vutils
from train_dcgan import G, NZ, CLASS_NAME

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = G().to(dev)
net.load_state_dict(torch.load(f"runs/gan/{CLASS_NAME}_G.pt", map_location=dev))
net.eval()

out = f"data/gan_synth/{CLASS_NAME}"
os.makedirs(out, exist_ok=True)
n = 0
with torch.no_grad():
    while n < 500:
        imgs = net(torch.randn(50, NZ, 1, 1, device=dev)).cpu()
        for im in imgs:
            vutils.save_image(im, f"{out}/{CLASS_NAME}_syn_{n:04d}.png", normalize=True, value_range=(-1, 1))
            n += 1
print("saved", n, "images in", out)