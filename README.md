# Gradient Regularization for Robust Deepfake Detection

An implementation of a deepfake detection model that uses gradient regularization to improve robustness against adversarial attacks. This approach perturbs the mean and standard deviation of shallow layers in an EfficientNetB0 backbone to enhance generalization and defend against attacks like FGSM and PGD.

## Overview

Traditional deepfake detectors are vulnerable to adversarial attacks that can fool them with imperceptible perturbations. This project implements a gradient regularization technique that trains the model on both original and adversarially perturbed features, making it more robust while maintaining high accuracy on clean data.

### Key Features

- **Gradient Regularization**: Combines gradients from original and perturbed data for robust training
- **Feature-Level Perturbations**: Targets shallow layer statistics (mean and standard deviation) rather than input pixels
- **EfficientNetB0 Backbone**: Leverages pre-trained EfficientNet for feature extraction
- **Mixed Precision Training**: Uses automatic mixed precision for efficient training
- **Adversarial Robustness**: Designed to defend against attacks like FGSM, and PGD. 

## Architecture

The system consists of three main components:

1. **DeepfakeDetector**: EfficientNetB0-based model with separate shallow and deep feature extractors
2. **PerturbationInjectionModule (PIM)**: Applies calculated perturbations to shallow features
3. **Gradient Regularization Training Loop**: Implements the dual-gradient optimization strategy

## Training Pipeline

The training process follows a sophisticated 5-step pipeline:

### 1. Initialization
- Clear existing gradients from the optimizer for each batch

### 2. Pass 1: Gradient from Original Data (g₁)
- Forward pass with original, unperturbed input data
- Calculate standard empirical loss L_original
- Backward pass to compute:
  - Gradients w.r.t. model parameters (stored as g₁)
  - Gradients w.r.t. shallow feature statistics (μₛ, σₛ)

### 3. Perturbation Calculation
- Use feature statistics gradients to calculate perturbation vector
- Apply small step of size `r` in gradient direction:
  ```
  δₗ = r × ∇L/||∇L||
  ```

### 4. Pass 2: Gradient from Perturbed Data (g₂)
- Apply perturbations to shallow features using PIM
- Forward pass with perturbed features
- Calculate perturbed loss L_perturbed
- Backward pass to compute gradients (stored as g₂)

### 5. Combine Gradients and Update
- Combine gradients using balance coefficient α:
  ```
  g_final = (1 - α) × g₁ + α × g₂
  ```
- Update model parameters with combined gradients

## Dataset

This project uses a combination of two major deepfake detection datasets:

- **DFFD (Diverse Fake Face Dataset)**: A comprehensive dataset containing various types of synthetic face images
- **FaceForensics++**: Extracted frames from video sequences containing both real and manipulated faces using different generation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures)

The combined dataset provides diverse training examples across different deepfake generation techniques, improving the model's generalization capabilities.

### Dataset Preparation

1. Download DFFD dataset from the official source
2. Download FaceForensics++ dataset and extract frames from videos
3. Combine and organize the datasets with proper train/validation splits
4. Apply data augmentation with random transformations (e.g., rotations, flips, color jitter) to increase diversity and mitigate class imbalance



## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `R_SCALAR` | 0.05 | Perturbation magnitude controlling adversarial strength |
| `ALPHA_COEFF` | 1.0 | Balance between original (g₁) and perturbed (g₂) gradients |
| `LEARNING_RATE` | 0.001 | SGD learning rate |
| `EPOCHS` | 10 | Number of training epochs |

## Model Checkpoints

The training process automatically saves checkpoints:
- Every 2 epochs: `grad_reg/model_epoch_{epoch}.pth`
- Final model: `grad_reg/final-full.pth`

## Key Equations

**Feature Normalization (Equation 6):**

![Equation 6](https://latex.codecogs.com/png.image?\dpi{115}f_{\theta_{s}}(x)_{\text{norm}}=\frac{f_{\theta_{s}}(x)-\mu_{s}}{\sigma_{s}})


**Perturbation Calculation (Equation 9):**

$$
\Delta l = r \frac{\nabla_{\mu_{s},\sigma_{s}}L(f_{\theta_{s}}(x),y,\theta_{d})}{\|\nabla_{\mu_{s},\sigma_{s}}L(f_{\theta_{s}}(x),y,\theta_{d})\|_{2}}
$$


**Combined Loss (Equation 12):**

![Equation 12](https://latex.codecogs.com/png.image?\dpi{115}\min_{\theta}(1-\alpha)L(x,y,\theta)%20+%20\alpha%20L(f'_{\theta_{s}}(x),y,\theta_{d}))


## Adversarial Robustness

This approach provides enhanced robustness against:
- **FGSM (Fast Gradient Sign Method)**: Single-step gradient-based attacks
- **PGD (Projected Gradient Descent)**: Multi-step iterative attacks

## Conclusion
The deepfake detection model, enhanced by gradient regularization, achieved significant robustness against adversarial attacks, with accuracy improvements to 33.23% at ε=0.001 FGSM (vs. 25.21% baseline) and 10.27% at ε=0.020 FGSM (vs. 5.53% baseline).

An ablation study optimized the selection of perturbation layers and fine-tuned hyperparameters (balance coefficient α and approximation scalar r), yielding a high-performing configuration. 

## Dataset Structure

```
dataset_combined/
├── train/
│   ├── fake/
│   │   ├── 000_face_1.png
│   │   ├── 001_face_1.png
│   │   └── ...
│   └── real/
│       ├── 000_face_1.png
│       ├── 001_face_1.png
│       └── ...
├── validation/
│   ├── fake/
│   └── real/
└── test/
    ├── fake/
    └── real/
```

## References

1. W. Guan, W. Wang, J. Dong and B. Peng, (2024). Improving Generalization of Deepfake Detectors by Imposing Gradient Regularization, In IEEE Transactions on Information Forensics and Security, vol. 19, pp. 5345-5356.

2. M. Tan and Q. Le, (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In Proc. Int. Conf. Mach. Learn., pp. 6105–6114.

3. H. Dang, F. Liu, J. Stehouwer, X. Liu, A. Jain, (2020). On the Detection of Digital Face Manipulation. In Proceedings of IEEE Computer Vision and Pattern Recognition (CVPR 2020), Seattle, WA, Jun. 2020.

4. M. Abbasi, P. Váz, J. Silva and P. Martins, (2025). Comprehensive Evaluation of Deepfake Detection Models: Accuracy, Generalization, and Resilience to Adversarial Attacks. Applied Sciences, 15(3), 1225.
