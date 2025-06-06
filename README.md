# GAN-Image-Style-Transfer


---

## Project Summary

This repository hosts the implementation for the Monet Style Transfer Competition, where the objective was to build a Generative Adversarial Network (GAN) to transform real-world photographs into Monet-style paintings and vice versa.

Using a CycleGAN architecture, our solution achieved high-quality style transfer. Performance was evaluated using Fréchet Inception Distance (FID) and Memorization-informed Fréchet Inception Distance (MiFID).

Leaderboard Result:  
2nd Place on Kaggle  
FID: 51.6575  
MiFID: 0.2203

We generated 933 Monet-style images from 7,038 photographs and 260 Monet paintings, submitted as `Team_28.zip` with `submission.csv`.

---

## Generative Adversarial Networks (GANs)

### Definition

A Generative Adversarial Network (GAN) is a deep learning model comprising two components trained in a competitive setup:

- Generator: Creates synthetic data (e.g., Monet-style images).
- Discriminator: Evaluates whether input data is real or generated.

The adversarial training leads to the Generator producing highly realistic outputs over time.

---

## CycleGAN Overview

We used CycleGAN, designed for unpaired image-to-image translation, enabling transformation between two domains (photos ↔ Monet paintings) without paired samples. It uses cycle-consistency loss to ensure that the style translations are reversible.

---

## CycleGAN Architecture

### Components

#### Generators

- G_photo_to_monet: Converts photos to Monet-style paintings  
- G_monet_to_photo: Converts Monet paintings to realistic photos  

Architecture:
- ResNet-based with 9 residual blocks
- Initial convolution layers for encoding
- Residual blocks for feature transformation
- Transposed convolutions for decoding
- Input/Output: 256x256 RGB images normalized to [-1, 1]

#### Discriminators

- D_monet: Distinguishes real vs. generated Monet paintings  
- D_photo: Distinguishes real vs. generated photographs  

Architecture:
- PatchGAN (70x70 patches)
- Series of convolution layers with increasing filters
- Outputs a probability map indicating real/fake

---

## Loss Functions

- Adversarial Loss: Mean Squared Error (MSE) to train Discriminator and Generator
- Cycle-Consistency Loss: L1 loss between input and reconstructed image  
  λ_cycle = 10.0
- Total Generator Loss: Adversarial Loss + Cycle-Consistency Loss

---

## Optimization

- Optimizer: Adam (lr=0.0002, betas=(0.5, 0.999))
- Scheduler: Linear decay starting at epoch 100 via LambdaLR

---

## Training Details

### Dataset

- Monet Paintings: 260 images  
- Photographs: 7,038 images  
- Preprocessing:  
  - Resize to 256x256  
  - Normalize to [-1, 1] using torchvision.transforms

### Training Process

- Trained for 200 epochs
- Used separate DataLoaders for photos and paintings
- Discriminators trained to classify real vs. fake
- Generators optimized to fool Discriminators and preserve cycle-consistency
- Tracked average losses per epoch using tqdm

### Output

- 933 Monet-style images generated  
- Saved as 256x256 RGB JPGs in `generated_monet_images/`

---

## Kaggle Performance

Team_28 Submission:

- FID: 51.6575
- MiFID: 0.2203

These scores reflect both visual fidelity and diversity, as measured by the competition's official evaluation script.

---

## Evaluation Metrics

### Fréchet Inception Distance (FID)

- Definition: Measures the distance between feature distributions of real and generated images using Inception V3
- Formula:  
  FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^(1/2))
- Interpretation: Lower FID indicates better visual similarity
- Result: 51.6575

### Memorization-informed FID (MiFID)

- Definition: Adds a memorization penalty to FID, encouraging diverse, novel generations
- Interpretation: Lower MiFID implies better generalization
- Result: 0.2203

---

## Implementation Summary

### Data Preparation

- Custom Dataset class with torchvision.transforms
- Separate DataLoaders for Monet and photo datasets

### Model Development

- Implemented CycleGAN in PyTorch:
  - ResNet-based Generators
  - PatchGAN Discriminators
- Custom training loop with MSE + L1 losses

### Training

- 200 epochs
- Model checkpoints every 50 epochs
- Final model saved as `cyclegan_model.pth`

### Image Generation

- Generated Monet-style images using G_photo_to_monet
- Saved in `generated_monet_images/`
- Archived as `Team_28.zip`

### Submission

- `submission.csv` with FID and MiFID scores
- `Team_28.zip` with 933 images

---



