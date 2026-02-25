import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import normal_ad


def _fit_statsmodels(x, y):
    x_const = sm.add_constant(x)
    return sm.OLS(y, x_const).fit()


def _residuals(model_sm, x, y):
    x_const = sm.add_constant(x)
    y_pred = model_sm.predict(x_const)
    return np.array(y) - np.array(y_pred)


def check_linearity(model_sm, p_value_thresh=0.05):
    p_value = model_sm.f_pvalue
    passed = p_value < p_value_thresh
    status = "OK  linearna veza" if passed else "X linearna veza nije potvrdjena"
    print(f"  Linearnost (F-test)       : p = {p_value:.4e} -> {status}")
    return passed, p_value


def check_independence(residuals):
    dw = durbin_watson(residuals)
    passed = 1.5 <= dw <= 2.5
    status = "OK  nema autokorelacije" if passed else "X postoji autokorelacija"
    print(f"  Nezavisnost gresaka (DW)  : d = {dw:.3f}     -> {status}")
    return passed, dw


def check_normality(residuals, p_value_thresh=0.05):
    p_value = normal_ad(residuals)[1]
    passed = p_value >= p_value_thresh
    status = "OK  normalna raspodela" if passed else "X nije normalna raspodela"
    print(f"  Normalnost reziduala (AD) : p = {p_value:.4f}   -> {status}")
    return passed, p_value


def check_homoscedasticity(residuals, x, p_value_thresh=0.05):
    x_const = sm.add_constant(x)
    p_value = sm.stats.het_goldfeldquandt(residuals, x_const)[1]
    passed = p_value >= p_value_thresh
    status = "OK  jednaka varijansa" if passed else "X nejednaka varijansa"
    print(f"  Homoskedasticnost (GQ)    : p = {p_value:.4f}   -> {status}")
    return passed, p_value


def check_collinearity(x, thresh=0.999):
    corr = pd.DataFrame(x).corr().values
    np.fill_diagonal(corr, np.nan)
    passed = not (np.nanmax(np.abs(corr)) > thresh)
    status = "OK  nema savrsene kolinearnosti" if passed else "X postoji savrsena kolinearnost"
    print(f"  Multikolinearnost         :             -> {status}")
    return passed


def validate_ols(x_train, y_train):
    print("=" * 57)
    print("PROVERA PRETPOSTAVKI OLS MODELA")
    print("=" * 57)

    model_sm = _fit_statsmodels(x_train, y_train)
    residuals = _residuals(model_sm, x_train, y_train)

    lin_ok, _ = check_linearity(model_sm)
    ind_ok, _ = check_independence(residuals)
    nor_ok, _ = check_normality(residuals)
    hom_ok, _ = check_homoscedasticity(residuals, x_train)
    col_ok    = check_collinearity(x_train)

    all_ok = all([lin_ok, ind_ok, nor_ok, hom_ok, col_ok])
    print("-" * 57)
    print(f"  Model validan: {'DA' if all_ok else 'NE -- neke pretpostavke nisu ispunjene'}")
    print("=" * 57)

    return all_ok


if __name__ == "__main__":
    from preprocessing import load_and_prepare

    x_train, x_val, x_test, y_train, y_val, y_test = load_and_prepare(cap=True)
    validate_ols(x_train, y_train)
