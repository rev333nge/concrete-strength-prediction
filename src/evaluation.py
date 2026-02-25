import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred):
    mask = np.array(y_true) != 0
    return np.mean(np.abs((np.array(y_true)[mask] - np.array(y_pred)[mask]) / np.array(y_true)[mask])) * 100


def evaluate_model(model, x_train, x_val, x_test, y_train, y_val, y_test):
    results = {}
    for split_name, x, y in [("train", x_train, y_train), ("val", x_val, y_val), ("test", x_test, y_test)]:
        y_pred = model.predict(x)
        results[split_name] = {
            "RMSE": round(rmse(y, y_pred), 2),
            "MAE": round(mae(y, y_pred), 2),
            "MAPE": round(mape(y, y_pred), 2),
        }
    return results


def ols_summary(x_train, y_train):
    x_with_const = sm.add_constant(x_train)
    model = sm.OLS(y_train, x_with_const).fit()
    print(model.summary())


def comparison_table(results_dict):
    rows = []
    for model_name, splits in results_dict.items():
        for split_name, metrics in splits.items():
            rows.append({"Model": model_name, "Split": split_name, **metrics})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from preprocessing import load_data, add_features, cap_outliers, MODEL_FEATURE_COLS, TARGET_COL, load_and_prepare
    from models import train_ols, tune_rf, tune_xgb
    from visualization import plot_actual_vs_predicted, plot_residuals, plot_feature_importance

    df = load_data()
    df = add_features(df)
    df = cap_outliers(df)
    print("OLS Summary — ceo dataset (statsmodels):")
    ols_summary(df[MODEL_FEATURE_COLS], df[TARGET_COL])

    x_train, x_val, x_test, y_train, y_val, y_test = load_and_prepare(cap=True)
    ols = train_ols(x_train, y_train)

    x_train_r, x_val_r, x_test_r, y_train_r, y_val_r, y_test_r = load_and_prepare(cap=False)

    print("Tuning RF...")
    rf = tune_rf(x_train_r, y_train_r)

    print("Tuning XGBoost...")
    xgb = tune_xgb(x_train_r, y_train_r, x_val_r, y_val_r)

    results = {
        "OLS": evaluate_model(ols, x_train, x_val, x_test, y_train, y_val, y_test),
        "RF":  evaluate_model(rf, x_train_r, x_val_r, x_test_r, y_train_r, y_val_r, y_test_r),
        "XGB": evaluate_model(xgb, x_train_r, x_val_r, x_test_r, y_train_r, y_val_r, y_test_r),
    }

    print(comparison_table(results).to_string(index=False))

    models_dict = {"OLS": ols, "RF": rf, "XGB": xgb}
    splits_dict = {
        "OLS": (x_test, y_test),
        "RF":  (x_test_r, y_test_r),
        "XGB": (x_test_r, y_test_r),
    }

    plot_actual_vs_predicted(models_dict, splits_dict)
    plot_residuals(models_dict, splits_dict)
    plot_feature_importance(rf, xgb, ols)

    r2_score = xgb.score(x_test_r, y_test_r)
    print(f"XGBoost R2 Score: {round(r2_score, 4)}")
