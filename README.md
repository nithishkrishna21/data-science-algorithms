# Algorithms for Data Science

A collection of applied data-science and machine-learning analyses, with an emphasis on
**implementing core algorithms from scratch** (not just calling library functions) and on
comparing methods on real datasets.

Each subfolder is a self-contained project with its own README and a single, runnable
notebook. All datasets are fetched at runtime (scikit-learn / OpenML / public URLs) — no
local data files required.

## Projects

| Project | Focus | Highlights |
|---|---|---|
| [eda-pca-naive-bayes-optimization](eda-pca-naive-bayes-optimization/) | Exploratory analysis + classical algorithms | Mahalanobis outlier detection, PCA, **Naïve Bayes from scratch**, Linear Programming vs Particle Swarm Optimization, Bayesian networks for inference |
| [mnist-dct-eigen-classification](mnist-dct-eigen-classification/) | Feature extraction, classification & generative models | 2D **Discrete Cosine Transform** features, **eigen-decomposition** dimensionality reduction, **SVM from scratch**, Random Forest, a PyTorch **CNN**, a Design-of-Experiments study (ANOVA + logistic regression), and a survey of **GANs/VAEs/Seq2Seq** (T5, Stable Diffusion) |
| [regression-svm-kmeans](regression-svm-kmeans/) | Supervised + unsupervised learning | Multicollinearity (VIF) + **Ridge/Lasso** regression, **SVM via quadratic programming**, and **K-Means from scratch** |

## Skills demonstrated

From-scratch algorithm implementation (Naïve Bayes, SVM/QP, K-Means, PSO), exploratory
data analysis, dimensionality reduction (PCA, eigen-decomposition), feature engineering
(DCT, polynomial features), regularization (Ridge/Lasso), probabilistic graphical models
(Bayesian networks), optimization (linear programming, particle swarm), model comparison
and cross-validation, and deep learning (CNN) — using NumPy, pandas, scikit-learn, SciPy,
statsmodels, NetworkX, and PyTorch.

## Running

Each project's notebook is standalone. From a project folder:

```bash
pip install numpy pandas scikit-learn scipy statsmodels matplotlib networkx torch
jupyter notebook analysis.ipynb
```

(Some projects use a subset of these; see each project's README.)

## Attribution

These were assignments for a graduate Algorithms for Data Science course; the analysis,
from-scratch implementations, and write-ups are the author's. Datasets are public
(UCI Wine Quality, MNIST via OpenML, California Housing, Breast Cancer Wisconsin, UCI
Wholesale Customers).