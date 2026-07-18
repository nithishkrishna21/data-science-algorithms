# MNIST Feature Extraction, Classification & Generative Models

Image classification on **MNIST** built around hand-engineered frequency features (from
classical methods to a CNN), a Design-of-Experiments study on a line-of-sight detection
simulation, and a comparative survey of modern generative sequence architectures.

## What's inside (`analysis.ipynb`)

**1. Feature extraction & classification (MNIST)**
- Preprocessing and visualization of the handwritten-digit images.
- **2D Discrete Cosine Transform (DCT)** applied to each image, with **directional frequency
  masks** (horizontal / vertical / diagonal) to select informative coefficients.
- **Eigen-decomposition** for dimensionality reduction over the extracted features.
- Classification comparisons:
  - **SVM implemented from scratch** vs. a Random Forest baseline (accuracy and training time).
  - A **PyTorch CNN** vs. the DCT-feature models — with analysis of *why* learned convolutional
    features outperform fixed DCT features.

**2. Design of Experiments (line-of-sight detection)**
- A structured DoE on a physics simulation of air-to-ground **line-of-sight (LOS)** detection:
  hypothesis, a Random Forest classifier, and **ANOVA + logistic regression** (`statsmodels`),
  reconciling feature importances against statistical significance.

**3. Generative models & sequence architectures**
- A comparative study of **GANs, VAEs, and Seq2Seq** models, demonstrated with pre-trained
  **T5** (text-to-text) and **Stable Diffusion** (text-to-image): architectures, loss functions,
  training challenges, and trade-offs.

## Key takeaways (from the notebook's analysis)
- Random Forest beat the from-scratch SVM on both accuracy and training time here (ensemble
  flexibility vs. a single model).
- The CNN outperformed DCT-based pipelines because its convolutional layers *learn* features
  rather than relying on a fixed transform basis.

## Stack
NumPy, pandas, scikit-learn, SciPy, statsmodels, **PyTorch**, matplotlib.

## Run
```bash
pip install numpy pandas scikit-learn scipy statsmodels torch matplotlib
jupyter notebook analysis.ipynb
```
MNIST is fetched via `sklearn.datasets.fetch_openml('mnist_784')` at runtime.