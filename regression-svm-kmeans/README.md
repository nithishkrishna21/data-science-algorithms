# Regression, SVM Classification & K-Means

Three supervised- and unsupervised-learning tasks, each pairing library tools with
from-scratch implementation and careful analysis.

## What's inside (`analysis.ipynb`)

**1. Regression with regularization (California Housing)**
- EDA on feature distributions and scales.
- Diagnoses **multicollinearity** via correlation heatmaps and **Variance Inflation Factors**
  (latitude/longitude and rooms/bedrooms are highly collinear).
- Trains Linear Regression, then **Ridge and Lasso** to mitigate multicollinearity, comparing
  how each shrinks or eliminates coefficients and their effect on predictive accuracy.

**2. Classification with SVMs (Breast Cancer Wisconsin)**
- Preprocessing and **feature engineering** (including polynomial features).
- **Support Vector Machine implemented from scratch via quadratic programming**, alongside
  library models.
- Model comparison with cross-validation: best overall (SVM with RBF kernel), most stable
  across folds (linear SVM), and the effect of kernel choice.

**3. K-Means clustering from scratch (UCI Wholesale Customers)**
- **K-Means implemented from scratch** (no scikit-learn): random centroid init, Euclidean
  assignment, centroid recomputation, convergence.
- A framework that evaluates multiple candidate cluster counts for customer segmentation.

## Stack
NumPy, pandas, scikit-learn, statsmodels, SciPy, matplotlib.

## Run
```bash
pip install numpy pandas scikit-learn statsmodels scipy matplotlib
jupyter notebook analysis.ipynb
```
Datasets: California Housing and Breast Cancer are built into scikit-learn; Wholesale
Customers is loaded from the UCI repository URL at runtime.