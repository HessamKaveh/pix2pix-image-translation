"""Fast CPU tests: U-Net / PatchGAN shapes, dataset pairing, uint8 conversion, PSNR/SSIM evaluation loop."""
import numpy as np
import pytest
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import evaluate
from dataset import Facades
from model import PatchDiscriminator, UNetGenerator, init_weights

torch.manual_seed(0)


def test_unet_generator_shape_and_range():
    G = UNetGenerator().eval()
    with torch.no_grad():
        y = G(torch.randn(2, 3, 256, 256))
    assert y.shape == (2, 3, 256, 256)
    assert y.min() >= -1 and y.max() <= 1                          # tanh output


def test_patchgan_receptive_grid():
    D = PatchDiscriminator()
    out = D(torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256))
    assert out.shape == (1, 1, 30, 30)                              # 70x70 PatchGAN on 256x256


def test_init_weights_distribution():
    G = UNetGenerator(ngf=16)
    G.apply(init_weights)
    w = torch.cat([m.weight.flatten() for m in G.modules() if isinstance(m, torch.nn.Conv2d)])
    assert abs(w.mean().item()) < 2e-3 and w.std().item() == pytest.approx(0.02, rel=0.1)


def test_facades_pairing_and_normalisation(tmp_path):
    for split in ("train", "test"):
        (tmp_path / split).mkdir()
        ab = np.zeros((256, 512, 3), np.uint8)
        ab[:, :256] = 255                                           # left half = photo (white)
        Image.fromarray(ab).save(tmp_path / split / "1.jpg")
    label, photo, name = Facades(root=str(tmp_path), split="test")[0]
    assert name == "1.jpg" and label.shape == photo.shape == (3, 256, 256)
    assert photo.mean() > 0.95 and label.mean() < -0.95             # B (label, right) -> A (photo, left)
    label, photo, _ = Facades(root=str(tmp_path), split="train")[0]
    assert label.shape == (3, 256, 256)                             # 286 resize + random 256 crop


def test_to_uint8_mapping():
    t = torch.tensor([-2.0, -1.0, 0.0, 1.0, 3.0]).view(1, 1, 5).expand(3, 1, 5)
    np.testing.assert_array_equal(evaluate.to_uint8(t)[0, :, 0], [0, 0, 128, 255, 255])


def test_psnr_ssim_reference_values():
    rng = np.random.RandomState(0)
    r = rng.randint(0, 256, (64, 64, 3)).astype(np.uint8)
    assert structural_similarity(r, r, channel_axis=2, data_range=255) == pytest.approx(1.0)
    f = np.clip(r.astype(int) + 10, 0, 255).astype(np.uint8)
    mse = np.mean((f.astype(float) - r) ** 2)
    assert peak_signal_noise_ratio(r, f, data_range=255) == pytest.approx(10 * np.log10(255 ** 2 / mse))


def test_evaluation_loop_writes_outputs(tmp_path, monkeypatch):
    (tmp_path / "checkpoints" / "l1").mkdir(parents=True)
    torch.save(UNetGenerator().state_dict(), tmp_path / "checkpoints" / "l1" / "generator.pt")
    monkeypatch.setattr(evaluate, "ROOT", str(tmp_path))
    loader = [(torch.rand(1, 3, 256, 256) * 2 - 1, torch.rand(1, 3, 256, 256) * 2 - 1, [f"{i}.jpg"]) for i in (1, 2)]
    m, outs = evaluate.run("l1", torch.device("cpu"), loader, str(tmp_path / "gen"))
    assert set(m) == {"L1", "PSNR", "SSIM"} and 0 < m["L1"] < 1 and m["PSNR"] > 0
    assert sorted(p.name for p in (tmp_path / "gen").iterdir()) == ["1.png", "2.png"]
    assert outs["1.jpg"].shape == (256, 256, 3) and outs["1.jpg"].dtype == np.uint8
