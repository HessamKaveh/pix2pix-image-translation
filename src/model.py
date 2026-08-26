import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    """بلاک U-Net ساده"""
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1, up=False):
        super().__init__()
        if up:
            self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Generator(nn.Module):
    """سادگیِ Pix2Pix Generator"""
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        
        # Encoder
        self.enc1 = UNetBlock(in_ch, 32, up=False)  # 256 -> 128
        self.enc2 = UNetBlock(32, 64, up=False)     # 128 -> 64
        self.enc3 = UNetBlock(64, 128, up=False)    # 64 -> 32
        self.enc4 = UNetBlock(128, 256, up=False)   # 32 -> 16
        
        # Bottleneck
        self.bottle = UNetBlock(256, 256, up=False)  # 16 -> 8
        
        # Decoder (بدون skip connections برای اجتناب مشکل کانال)
        self.dec4 = UNetBlock(256, 256, up=True)   # 8 -> 16
        self.dec3 = UNetBlock(256, 128, up=True)   # 16 -> 32
        self.dec2 = UNetBlock(128, 64, up=True)    # 32 -> 64
        self.dec1 = UNetBlock(64, 32, up=True)     # 64 -> 128
        self.dec0 = UNetBlock(32, 32, up=True)     # 128 -> 256
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(32, out_ch, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottle(e4)
        
        # Decoder (بدون concatenation)
        d4 = self.dec4(b)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        d0 = self.dec0(d1)
        
        # Output
        out = self.final(d0)
        return out


class PatchGANDiscriminator(nn.Module):
    """PatchGAN: تفکیک تصاویر واقعی از تولید شده"""
    def __init__(self, in_ch=6):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )
    
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.model(xy)
