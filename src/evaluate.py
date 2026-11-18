"""Evaluate trained generators on the 106-image Facades test split.

Metrics (computed on 256x256 outputs in [0, 255]):
  - L1 (mean absolute error, pixel range [0, 1])
  - PSNR and SSIM against the ground-truth photo
  - FID between generated and real test photos (torch-fidelity, Inception-v3 pool3)

Also writes a qualitative grid and the training curves.
"""
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

from dataset import Facades
from model import UNetGenerator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
MODES = {"l1": "L1 only", "gan_l1": "cGAN + L1 (pix2pix)"}


def to_uint8(t):
    return ((t.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def run(mode, dev, loader, gen_dir):
    G = UNetGenerator().to(dev)
    G.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", mode, "generator.pt"), map_location=dev))
    # pix2pix keeps dropout and uses batch statistics at test time (batch size 1) -> train() mode
    G.train()
    torch.manual_seed(0)
    os.makedirs(gen_dir, exist_ok=True)
    l1s, psnrs, ssims, outs = [], [], [], {}
    for label, photo, name in loader:
        fake = G(label.to(dev))
        f, r = to_uint8(fake[0]), to_uint8(photo[0])
        Image.fromarray(f).save(os.path.join(gen_dir, name[0].replace(".jpg", ".png")))
        l1s.append(np.abs(f.astype(np.float32) - r.astype(np.float32)).mean() / 255.0)
        psnrs.append(peak_signal_noise_ratio(r, f, data_range=255))
        ssims.append(structural_similarity(r, f, channel_axis=2, data_range=255))
        outs[name[0]] = f
    return {"L1": float(np.mean(l1s)), "PSNR": float(np.mean(psnrs)), "SSIM": float(np.mean(ssims))}, outs


def main():
    import torch_fidelity

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = Facades(split="test")
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    tmp = os.path.join(ROOT, "checkpoints", "eval_tmp")
    real_dir = os.path.join(tmp, "real")
    os.makedirs(real_dir, exist_ok=True)
    labels, reals = {}, {}
    for label, photo, name in loader:
        reals[name[0]] = to_uint8(photo[0]); labels[name[0]] = to_uint8(label[0])
        Image.fromarray(reals[name[0]]).save(os.path.join(real_dir, name[0].replace(".jpg", ".png")))

    metrics, outputs = {}, {}
    for mode in MODES:
        if not os.path.exists(os.path.join(ROOT, "checkpoints", mode, "generator.pt")):
            continue
        gen_dir = os.path.join(tmp, mode)
        m, outputs[mode] = run(mode, dev, loader, gen_dir)
        fid = torch_fidelity.calculate_metrics(input1=gen_dir, input2=real_dir, cuda=dev.type == "cuda",
                                               fid=True, verbose=False)
        m["FID"] = float(fid["frechet_inception_distance"])
        metrics[MODES[mode]] = m
        print(MODES[mode], {k: round(v, 4) for k, v in m.items()})

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # qualitative grid
    names = sorted(reals, key=lambda n: int(n.split(".")[0]))[::18][:6]
    cols = ["Input label map"] + [MODES[m] for m in outputs] + ["Ground truth"]
    fig, ax = plt.subplots(len(names), len(cols), figsize=(2.6 * len(cols), 2.6 * len(names)))
    for i, n in enumerate(names):
        row = [labels[n]] + [outputs[m][n] for m in outputs] + [reals[n]]
        for j, im in enumerate(row):
            ax[i, j].imshow(im); ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(cols[j], fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "facades_test_samples.png"), dpi=110)
    plt.close()

    # training curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for mode in MODES:
        p = os.path.join(RES, f"history_{mode}.json")
        if os.path.exists(p):
            h = json.load(open(p))
            ax[0].plot(h["g_l1"], label=MODES[mode])
            if mode == "gan_l1":
                ax[1].plot(h["g_gan"], label="G adversarial")
                ax[1].plot(h["d"], label="D")
    ax[0].set(title="Train L1 (generator)", xlabel="epoch"); ax[0].legend()
    ax[1].set(title="Adversarial losses (pix2pix)", xlabel="epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "training_curves.png"), dpi=110)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
