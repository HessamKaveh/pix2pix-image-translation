"""Train pix2pix on CMP Facades (label map -> photo).

Paper schedule: 200 epochs, batch size 1, Adam(2e-4, betas 0.5/0.999),
constant LR for 100 epochs then linear decay to 0, lambda_L1 = 100.

    python src/train.py --mode gan_l1      # full pix2pix (cGAN + L1)
    python src/train.py --mode l1          # ablation: L1 only (no discriminator)
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Facades
from model import PatchDiscriminator, UNetGenerator, init_weights

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gan_l1", "l1"], default="gan_l1")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--decay_from", type=int, default=100)
    ap.add_argument("--lambda_l1", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(ROOT, "checkpoints", args.mode)
    os.makedirs(out_dir, exist_ok=True)

    loader = DataLoader(Facades(split="train"), batch_size=1, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    G = UNetGenerator().to(dev); G.apply(init_weights)
    D = PatchDiscriminator().to(dev); D.apply(init_weights)
    opt_g = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    lr_lambda = lambda e: 1.0 - max(0, e - args.decay_from) / float(args.epochs - args.decay_from + 1)
    sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
    sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda)
    bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()
    use_gan = args.mode == "gan_l1"

    hist = {"g_gan": [], "g_l1": [], "d": []}
    t0 = time.time()
    for ep in range(args.epochs):
        G.train(); D.train()
        s = {"g_gan": 0.0, "g_l1": 0.0, "d": 0.0}
        for label, photo, _ in loader:
            label, photo = label.to(dev, non_blocking=True), photo.to(dev, non_blocking=True)
            fake = G(label)

            if use_gan:
                # D step: real pair -> 1, fake pair -> 0 (loss halved, as in the paper)
                pr, pf = D(label, photo), D(label, fake.detach())
                d_loss = 0.5 * (bce(pr, torch.ones_like(pr)) + bce(pf, torch.zeros_like(pf)))
                opt_d.zero_grad(set_to_none=True); d_loss.backward(); opt_d.step()
                s["d"] += d_loss.item()

            # G step
            g_l1 = l1(fake, photo)
            g_loss = args.lambda_l1 * g_l1
            if use_gan:
                pf = D(label, fake)
                g_gan = bce(pf, torch.ones_like(pf))
                g_loss = g_loss + g_gan
                s["g_gan"] += g_gan.item()
            opt_g.zero_grad(set_to_none=True); g_loss.backward(); opt_g.step()
            s["g_l1"] += g_l1.item()

        sch_g.step(); sch_d.step()
        for k in s:
            hist[k].append(s[k] / len(loader))
        print(f"[{args.mode}] epoch {ep + 1}/{args.epochs}  L1 {hist['g_l1'][-1]:.4f}  "
              f"G_gan {hist['g_gan'][-1]:.3f}  D {hist['d'][-1]:.3f}  "
              f"lr {sch_g.get_last_lr()[0]:.2e}  {(time.time() - t0) / 60:.1f} min", flush=True)
        if (ep + 1) % 50 == 0 or ep + 1 == args.epochs:
            torch.save(G.state_dict(), os.path.join(out_dir, "generator.pt"))

    torch.save(G.state_dict(), os.path.join(out_dir, "generator.pt"))
    if use_gan:
        torch.save(D.state_dict(), os.path.join(out_dir, "discriminator.pt"))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"history_{args.mode}.json"), "w") as f:
        json.dump(hist, f)


if __name__ == "__main__":
    main()
