# Concrete Strength Prediction

The project predicts the compressive strength of concrete (MPa) from its mix composition and age, using three models: OLS linear regression, Random Forest and XGBoost.

## Research questions

1. Which mix ingredients have the greatest effect on strength?
2. Is linear regression sufficient, or are nonlinear models necessary?
3. Which composition and age of concrete give the optimal predicted strength?

## Data

The dataset contains **1,030 samples** with 8 input features and a target variable:

| Feature | Description |
|---|---|
| Cement | kg/m³ |
| Blast Furnace Slag | kg/m³ |
| Fly Ash | kg/m³ |
| Water | kg/m³ |
| Superplasticizer | kg/m³ |
| Coarse Aggregate | kg/m³ |
| Fine Aggregate | kg/m³ |
| Age | days (1–365) |
| **Strength** | **MPa — target variable** |

Feature engineering adds `water_cement_ratio = Water / Cement` (Abrams' law).

## Model results (test set)

| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| OLS | 9.04 | 6.97 | 23.86% |
| Random Forest | 6.17 | 4.26 | 14.26% |
| **XGBoost** | **5.49** | **3.71** | **11.73%** |

XGBoost outperforms OLS by ~39% in RMSE. OLS was trained on the 6 optimal features (after ablation analysis), with outlier capping on the train set.

## Project structure

```
├── data/
│   └── concrete_data.csv      # dataset
├── src/
│   ├── preprocessing.py       # loading, feature engineering, data split
│   ├── models.py              # training OLS, RF, XGBoost (GridSearchCV)
│   ├── evaluation.py          # metrics (RMSE, MAE, MAPE) and pipeline entrypoint
│   ├── validation.py          # OLS model assumption checks
│   └── visualization.py       # plot generation
├── figures/                   # generated PNG plots
└── notebook.ipynb             # full analysis with narrative
```

## References

**Dataset**
- Tripathi, P. (2020). *Concrete Compressive Strength Dataset*. Kaggle.
  [https://www.kaggle.com/datasets/prathamtripathi/regression-with-neural-networking](https://www.kaggle.com/datasets/prathamtripathi/regression-with-neural-networking)

**Related work**
- Kadiwal, A. *EDA — Concrete Strength Prediction (R²=0.93)*. Kaggle Notebook.
  [https://www.kaggle.com/code/adityakadiwal/eda-concrete-strength-prediction-r2-0-93](https://www.kaggle.com/code/adityakadiwal/eda-concrete-strength-prediction-r2-0-93)

**Literature**
- Raschka, S., Liu, Y., & Mirjalili, V. (2022). *Machine Learning with PyTorch and Scikit-Learn*. Packt Publishing.
  [https://www.packtpub.com/product/machine-learning-with-pytorch-and-scikit-learn/9781801819312](https://www.packtpub.com/product/machine-learning-with-pytorch-and-scikit-learn/9781801819312)

**Abrams' law**
- Wikipedia: *Abrams' law — the relationship between the water-cement ratio and concrete strength*.
  [https://en.wikipedia.org/wiki/Abrams%27_law](https://en.wikipedia.org/wiki/Abrams%27_law)

## Running

```bash
pip install -r requirements.txt

# Full pipeline: train all three models + plots
python src/evaluation.py

# OLS assumption diagnostics
python src/validation.py

# Generate plots only
python src/visualization.py

# Interactive analysis
jupyter notebook notebook.ipynb
```
