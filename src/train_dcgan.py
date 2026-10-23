import os, torch, torch.nn as nn
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import sys
CLASS_NAME = sys.argv[1] if len(sys.argv) > 1 else "PET_Bottle"         # koi bhi class chuno
EPOCHS, BS, NZ = 100, 64, 100
OUT = "runs/gan"

def winit(m):
    n = m.__class__.__name__
    if "Conv" in n:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in n:
        nn.init.normal_(m.weight, 1.0, 0.02); nn.init.zeros_(m.bias)

class G(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.ConvTranspose2d(NZ, 512, 4, 1, 0, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Tanh())
    def forward(s, z): return s.net(z)

class D(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False))
    def forward(s, x): return s.net(x).view(-1)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                             transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    ds = datasets.ImageFolder(f"data/gan/{CLASS_NAME}", transform=tf)
    dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=2, drop_last=True)
    print("training images:", len(ds))

    netG, netD = G().to(dev), D().to(dev)
    netG.apply(winit); netD.apply(winit)
    bce = nn.BCEWithLogitsLoss()
    oG = torch.optim.Adam(netG.parameters(), 2e-4, betas=(0.5, 0.999))
    oD = torch.optim.Adam(netD.parameters(), 2e-4, betas=(0.5, 0.999))
    fixed = torch.randn(64, NZ, 1, 1, device=dev)
    vutils.save_image(next(iter(dl))[0][:64], f"{OUT}/{CLASS_NAME}_real.png", normalize=True, nrow=8)

    for ep in range(1, EPOCHS + 1):
        for real, _ in dl:
            real = real.to(dev); b = real.size(0)
            fake = netG(torch.randn(b, NZ, 1, 1, device=dev))
            lossD = (bce(netD(real), torch.ones(b, device=dev) * 0.9)
                     + bce(netD(fake.detach()), torch.zeros(b, device=dev)))
            oD.zero_grad(); lossD.backward(); oD.step()
            lossG = bce(netD(fake), torch.ones(b, device=dev))
            oG.zero_grad(); lossG.backward(); oG.step()
        print(f"epoch {ep}/{EPOCHS}  D={lossD.item():.3f}  G={lossG.item():.3f}")
        if ep == 1 or ep % 10 == 0:
            with torch.no_grad():
                vutils.save_image(netG(fixed).cpu(), f"{OUT}/{CLASS_NAME}_ep{ep:03d}.png",
                                  normalize=True, nrow=8)
    torch.save(netG.state_dict(), f"{OUT}/{CLASS_NAME}_G.pt")