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


def ols_ablation(x_train, x_val, y_train, y_val):
    from sklearn.linear_model import LinearRegression

    remaining = list(x_train.columns)
    results = []

    model = LinearRegression().fit(x_train[remaining], y_train)
    results.append({
        "step": 0,
        "n_features": len(remaining),
        "removed": None,
        "features": list(remaining),
        "train_rmse": round(rmse(y_train, model.predict(x_train[remaining])), 2),
        "val_rmse": round(rmse(y_val, model.predict(x_val[remaining])), 2),
    })

    step = 1
    while len(remaining) > 1:
        best_val_rmse = float("inf")
        best_feature = None

        for feat in remaining:
            subset = [f for f in remaining if f != feat]
            model = LinearRegression().fit(x_train[subset], y_train)
            val_rmse = rmse(y_val, model.predict(x_val[subset]))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_feature = feat

        remaining = [f for f in remaining if f != best_feature]
        model = LinearRegression().fit(x_train[remaining], y_train)
        results.append({
            "step": step,
            "n_features": len(remaining),
            "removed": best_feature,
            "features": list(remaining),
            "train_rmse": round(rmse(y_train, model.predict(x_train[remaining])), 2),
            "val_rmse": round(rmse(y_val, model.predict(x_val[remaining])), 2),
        })
        step += 1

    return results


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
    df, _ = cap_outliers(df)
    print("OLS Summary — ceo dataset (statsmodels):")
    ols_summary(df[MODEL_FEATURE_COLS], df[TARGET_COL])

    x_train, x_val, x_test, y_train, y_val, y_test = load_and_prepare(cap=True)

    # Ablacija za optimalne OLS feature-e
    ablation_results = ols_ablation(x_train, x_val, y_train, y_val)
    best_step = min(ablation_results, key=lambda r: r["val_rmse"])
    ols_features = best_step["features"]
    print(f"OLS optimalni feature-i ({len(ols_features)}): {ols_features}")

    ols = train_ols(x_train[ols_features], y_train)

    x_train_r, x_val_r, x_test_r, y_train_r, y_val_r, y_test_r = load_and_prepare(cap=False)

    print("Tuning RF...")
    rf = tune_rf(x_train_r, y_train_r)

    print("Tuning XGBoost...")
    xgb = tune_xgb(x_train_r, y_train_r)

    results = {
        "OLS": evaluate_model(ols, x_train[ols_features], x_val[ols_features], x_test[ols_features], y_train, y_val, y_test),
        "RF":  evaluate_model(rf, x_train_r, x_val_r, x_test_r, y_train_r, y_val_r, y_test_r),
        "XGB": evaluate_model(xgb, x_train_r, x_val_r, x_test_r, y_train_r, y_val_r, y_test_r),
    }

    print(comparison_table(results).to_string(index=False))

    models_dict = {"OLS": ols, "RF": rf, "XGB": xgb}
    splits_dict = {
        "OLS": (x_test[ols_features], y_test),
        "RF":  (x_test_r, y_test_r),
        "XGB": (x_test_r, y_test_r),
    }

    plot_actual_vs_predicted(models_dict, splits_dict)
    plot_residuals(models_dict, splits_dict)
    plot_feature_importance(rf, xgb, ols)

    r2_score = xgb.score(x_test_r, y_test_r)
    print(f"XGBoost R2 Score: {round(r2_score, 4)}")
