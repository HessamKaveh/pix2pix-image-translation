# Pix2Pix: Image-to-Image Translation with GAN

Conditional GAN (cGAN) for paired image-to-image translation.
Learns to transform sketch → realistic image using adversarial + L1 loss.

**Data note:** this project uses a small procedurally generated (synthetic) dataset built in `src/dataset.py` (300 paired sketch/colour images of simple shapes), not real images. It is a didactic implementation: the numbers below show that the pipeline trains, and are not comparable to benchmark results.

## Architecture
- **Generator**: U-Net with skip connections
- **Discriminator**: PatchGAN classifier
- **Loss**: Adversarial loss + L1 reconstruction loss

## Applications
- Sketch to photo
- Night to day (domain transfer)
- Semantic map to image
- Satellite → map imagery

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/train.py
python src/evaluate.py
```

## Results

| Metric | Value |
|---|---|
| Generator L1 loss (final epoch) | 0.266 |
| Epochs | 15 |

<p align="center"><img src="results/training_curves.png" width="85%"></p>

> The discriminator loss collapses towards 0 (0.002 at the last epoch), a sign that D overpowers G on this toy data.

## Author
Hessam Kaveh — Research Fellow, Italian Institute of Technology (IIT)

## License

Released under the [MIT License](LICENSE).
