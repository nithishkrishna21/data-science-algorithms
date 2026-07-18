# EDA, PCA, Naïve Bayes, Optimization & Bayesian Networks

A broad tour of classical data-science algorithms on the **UCI Wine Quality** dataset and
beyond, mixing library-based analysis with from-scratch implementations.

## What's inside (`analysis.ipynb`)

**1. Exploratory data analysis & preprocessing (Wine Quality)**
- Summary statistics and data visualization.
- **Outlier detection and removal via Mahalanobis distance** (accounts for feature covariance).
- Feature scaling / normalization.
- **PCA** for dimensionality reduction, with analysis of explained variance.

**2. Naïve Bayes classifier from scratch**
- Bins wine quality into Low / Average / High classes.
- **Gaussian Naïve Bayes implemented without ML libraries** (priors + per-feature likelihoods).
- Runtime-complexity analysis of the training and inference steps.
- Comparison against a library baseline.

**3. Linear Programming vs. Particle Swarm Optimization**
- Solves the same optimization problem two ways: `scipy.optimize.linprog` and PSO.
- Compares solution quality, computational efficiency, and robustness (LP is fast and exact
  for linear problems; PSO generalizes to non-linear/non-convex ones).

**4. Bayesian networks for diagnosis**
- Constructs a Bayesian network for medical diagnosis and performs **exact and approximate
  probabilistic inference**.
- Analyzes how graph structure affects inference runtime.

## Stack
NumPy, pandas, scikit-learn, SciPy, NetworkX, matplotlib.

## Run
```bash
pip install numpy pandas scikit-learn scipy networkx matplotlib
jupyter notebook analysis.ipynb
```
Data (UCI Wine Quality) is downloaded from public URLs at runtime.