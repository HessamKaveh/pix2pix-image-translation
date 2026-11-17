import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import Generator, PatchGANDiscriminator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(batch_size=16)

    generator = Generator(in_ch=3, out_ch=3).to(device)
    discriminator = PatchGANDiscriminator(in_ch=6).to(device)

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    optimizer_g = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    num_epochs = 15
    history = {"g_loss": [], "d_loss": [], "l1_loss": []}
    best_l1 = float("inf")

    lambda_l1 = 100  # L1 loss weight

    for epoch in range(num_epochs):
        g_loss_sum = 0.0
        d_loss_sum = 0.0
        l1_loss_sum = 0.0

        for sketch, real in train_loader:
            sketch, real = sketch.to(device), real.to(device)

            # Generator
            generated = generator(sketch)

            # Discriminator
            fake_pair = torch.cat([sketch, generated], dim=1)
            real_pair = torch.cat([sketch, real], dim=1)

            fake_pred = discriminator(sketch, generated.detach())
            real_pred = discriminator(sketch, real)

            # Loss
            d_loss = criterion_gan(fake_pred, torch.zeros_like(fake_pred)) + \
                     criterion_gan(real_pred, torch.ones_like(real_pred))

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # Generator
            fake_pred = discriminator(sketch, generated)
            g_gan_loss = criterion_gan(fake_pred, torch.ones_like(fake_pred))
            g_l1_loss = criterion_l1(generated, real)
            g_loss = g_gan_loss + lambda_l1 * g_l1_loss

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            g_loss_sum += g_loss.item() * sketch.size(0)
            d_loss_sum += d_loss.item() * sketch.size(0)
            l1_loss_sum += g_l1_loss.item() * sketch.size(0)

        g_loss_avg = g_loss_sum / len(train_loader.dataset)
        d_loss_avg = d_loss_sum / len(train_loader.dataset)
        l1_loss_avg = l1_loss_sum / len(train_loader.dataset)

        history["g_loss"].append(g_loss_avg)
        history["d_loss"].append(d_loss_avg)
        history["l1_loss"].append(l1_loss_avg)

        print(f"Epoch {epoch+1}/{num_epochs} | G Loss: {g_loss_avg:.4f} | D Loss: {d_loss_avg:.4f} | L1: {l1_loss_avg:.4f}")

        if l1_loss_avg < best_l1:
            best_l1 = l1_loss_avg
            torch.save(generator.state_dict(), os.path.join(RESULTS_DIR, "best_generator.pt"))

    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["g_loss"])
    axes[0].set_title("Generator Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(history["d_loss"])
    axes[1].set_title("Discriminator Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    axes[2].plot(history["l1_loss"])
    axes[2].set_title("L1 Loss (Sketch → Real)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"))
    print(f"\nBest L1 loss: {best_l1:.4f}")


if __name__ == "__main__":
    train()
