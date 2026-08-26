# Pix2Pix: Image-to-Image Translation with GAN

Conditional GAN (cGAN) for paired image-to-image translation.
Learns to transform sketch → realistic image using adversarial + L1 loss.

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
- `results/training_curves.png` — G/D loss curves
- `results/pix2pix_results.png` — sketch → generated → real comparison

## Author
Hessam Kaveh — Research Fellow, Italian Institute of Technology
