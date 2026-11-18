"""pix2pix networks (Isola et al., CVPR 2017).

Generator: U-Net-256 (8 down / 8 up, skip connections, dropout in the 3 innermost decoder blocks).
Discriminator: 70x70 PatchGAN (C64-C128-C256-C512), conditioned on the input image.
"""
import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Down(nn.Module):
    def __init__(self, cin, cout, norm=True):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(cout))
        layers.append(nn.LeakyReLU(0.2, True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, cin, cout, dropout=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout)]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(True))
        self.net = nn.Sequential(*layers)

    def forward(self, x, skip):
        return torch.cat([self.net(x), skip], 1)


class UNetGenerator(nn.Module):
    def __init__(self, cin=3, cout=3, ngf=64):
        super().__init__()
        f = ngf
        self.d1 = Down(cin, f, norm=False)       # 128
        self.d2 = Down(f, 2 * f)                 # 64
        self.d3 = Down(2 * f, 4 * f)             # 32
        self.d4 = Down(4 * f, 8 * f)             # 16
        self.d5 = Down(8 * f, 8 * f)             # 8
        self.d6 = Down(8 * f, 8 * f)             # 4
        self.d7 = Down(8 * f, 8 * f)             # 2
        self.bottleneck = nn.Sequential(nn.Conv2d(8 * f, 8 * f, 4, 2, 1), nn.ReLU(True))  # 1
        self.u1 = Up(8 * f, 8 * f, dropout=True)
        self.u2 = Up(16 * f, 8 * f, dropout=True)
        self.u3 = Up(16 * f, 8 * f, dropout=True)
        self.u4 = Up(16 * f, 8 * f)
        self.u5 = Up(16 * f, 4 * f)
        self.u6 = Up(8 * f, 2 * f)
        self.u7 = Up(4 * f, f)
        self.out = nn.Sequential(nn.ConvTranspose2d(2 * f, cout, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.d1(x); e2 = self.d2(e1); e3 = self.d3(e2); e4 = self.d4(e3)
        e5 = self.d5(e4); e6 = self.d6(e5); e7 = self.d7(e6)
        b = self.bottleneck(e7)
        y = self.u1(b, e7); y = self.u2(y, e6); y = self.u3(y, e5); y = self.u4(y, e4)
        y = self.u5(y, e3); y = self.u6(y, e2); y = self.u7(y, e1)
        return self.out(y)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN: outputs a 30x30 map of real/fake logits for a 256x256 input."""

    def __init__(self, cin=6, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, 2 * ndf, 4, 2, 1, bias=False), nn.BatchNorm2d(2 * ndf), nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * ndf, 4 * ndf, 4, 2, 1, bias=False), nn.BatchNorm2d(4 * ndf), nn.LeakyReLU(0.2, True),
            nn.Conv2d(4 * ndf, 8 * ndf, 4, 1, 1, bias=False), nn.BatchNorm2d(8 * ndf), nn.LeakyReLU(0.2, True),
            nn.Conv2d(8 * ndf, 1, 4, 1, 1),
        )

    def forward(self, cond, img):
        return self.net(torch.cat([cond, img], 1))
