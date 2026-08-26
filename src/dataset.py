import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageDraw
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


class SyntheticPix2PixDataset(Dataset):
    """شبیه‌سازی: sketch → عکس (دیتاست ساخت شده)"""
    def __init__(self, num_samples=300, img_size=256):
        self.num_samples = num_samples
        self.img_size = img_size
        
        # ایجاد دیتاست
        self._create_dataset()
    
    def _create_dataset(self):
        """تولید تصاویر sketch و واقعی"""
        self.sketches = []
        self.reals = []
        
        for idx in range(self.num_samples):
            sketch, real = self._generate_pair(idx)
            self.sketches.append(sketch)
            self.reals.append(real)
    
    def _generate_pair(self, idx):
        """تولید جفت: sketch + تصویر واقعی"""
        np.random.seed(idx)
        random.seed(idx)
        
        # تصویر واقعی: رنگی
        real = Image.new('RGB', (self.img_size, self.img_size), color='white')
        draw_real = ImageDraw.Draw(real)
        
        # sketch: سیاه و سفید
        sketch = Image.new('RGB', (self.img_size, self.img_size), color='white')
        draw_sketch = ImageDraw.Draw(sketch)
        
        # رسم اشکال تصادفی
        num_shapes = random.randint(3, 6)
        
        for _ in range(num_shapes):
            shape_type = random.choice(['rect', 'circle', 'line'])
            x1 = random.randint(20, self.img_size - 100)
            y1 = random.randint(20, self.img_size - 100)
            
            if shape_type == 'rect':
                x2 = x1 + random.randint(30, 100)
                y2 = y1 + random.randint(30, 100)
                
                # Sketch: خط سیاه
                draw_sketch.rectangle([x1, y1, x2, y2], outline='black', width=2)
                
                # Real: مستطیل رنگی
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                draw_real.rectangle([x1, y1, x2, y2], fill=color, outline='black', width=2)
            
            elif shape_type == 'circle':
                r = random.randint(20, 50)
                
                draw_sketch.ellipse([x1, y1, x1+2*r, y1+2*r], outline='black', width=2)
                
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                draw_real.ellipse([x1, y1, x1+2*r, y1+2*r], fill=color, outline='black', width=2)
            
            elif shape_type == 'line':
                x2 = random.randint(20, self.img_size - 20)
                y2 = random.randint(20, self.img_size - 20)
                
                draw_sketch.line([x1, y1, x2, y2], fill='black', width=3)
                
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                draw_real.line([x1, y1, x2, y2], fill=color, width=3)
        
        # تبدیل به tensor
        sketch_tensor = torch.from_numpy(np.array(sketch).transpose(2, 0, 1)).float() / 255.0
        real_tensor = torch.from_numpy(np.array(real).transpose(2, 0, 1)).float() / 255.0
        
        return sketch_tensor, real_tensor
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.sketches[idx], self.reals[idx]


def get_dataloaders(batch_size=16):
    full_dataset = SyntheticPix2PixDataset(num_samples=300)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader
