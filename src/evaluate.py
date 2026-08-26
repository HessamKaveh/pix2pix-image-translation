import os
import torch
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import Generator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_generator.pt")


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = get_dataloaders(batch_size=6)

    generator = Generator(in_ch=3, out_ch=3).to(device)
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    generator.eval()

    sketches, reals = next(iter(val_loader))
    sketches = sketches.to(device)

    with torch.no_grad():
        generated = generator(sketches)

    n_samples = min(5, sketches.size(0))
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))

    for i in range(n_samples):
        sketch = sketches[i].cpu().permute(1, 2, 0).numpy()
        real = reals[i].permute(1, 2, 0).numpy()
        gen = generated[i].cpu().permute(1, 2, 0).numpy()

        axes[i, 0].imshow(sketch)
        axes[i, 0].set_title("Sketch (Input)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gen)
        axes[i, 1].set_title("Generated (Output)")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(real)
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis("off")

    plt.suptitle("Pix2Pix: Sketch → Real Translation")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pix2pix_results.png"), dpi=150)
    print(f"Saved Pix2Pix results to {os.path.join(RESULTS_DIR, 'pix2pix_results.png')}")


if __name__ == "__main__":
    evaluate()
