import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


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


def comparison_table(results_dict):
    rows = []
    for model_name, splits in results_dict.items():
        for split_name, metrics in splits.items():
            rows.append({"Model": model_name, "Split": split_name, **metrics})
    return pd.DataFrame(rows)
