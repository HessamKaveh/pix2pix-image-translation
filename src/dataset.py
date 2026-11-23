"""CMP Facades paired dataset (pix2pix release).

Each file is a 512x256 image: left half = photo (A), right half = label map (B).
We translate label map -> photo (B -> A), as in the pix2pix paper.
"""
import os
import random

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

DEFAULT_ROOT = os.environ.get(
    "FACADES_ROOT", os.path.expanduser("~/datasets/portfolio/facades/facades"))


class Facades(Dataset):
    def __init__(self, root=DEFAULT_ROOT, split="train", load_size=286, crop_size=256):
        self.dir = os.path.join(root, split)
        self.files = sorted(os.listdir(self.dir), key=lambda f: int(os.path.splitext(f)[0]))
        self.train = split == "train"
        self.load_size, self.crop_size = load_size, crop_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        ab = Image.open(os.path.join(self.dir, self.files[i])).convert("RGB")
        w, h = ab.size
        photo = ab.crop((0, 0, w // 2, h))
        label = ab.crop((w // 2, 0, w, h))

        if self.train:
            # random jitter: resize to 286, random 256 crop, random horizontal flip
            label = TF.resize(label, [self.load_size] * 2, TF.InterpolationMode.BICUBIC)
            photo = TF.resize(photo, [self.load_size] * 2, TF.InterpolationMode.BICUBIC)
            top = random.randint(0, self.load_size - self.crop_size)
            left = random.randint(0, self.load_size - self.crop_size)
            label = TF.crop(label, top, left, self.crop_size, self.crop_size)
            photo = TF.crop(photo, top, left, self.crop_size, self.crop_size)
            if random.random() < 0.5:
                label, photo = TF.hflip(label), TF.hflip(photo)
        else:
            label = TF.resize(label, [self.crop_size] * 2, TF.InterpolationMode.BICUBIC)
            photo = TF.resize(photo, [self.crop_size] * 2, TF.InterpolationMode.BICUBIC)

        to = lambda im: TF.normalize(TF.to_tensor(im), [0.5] * 3, [0.5] * 3)  # [-1, 1]
        return to(label), to(photo), self.files[i]
