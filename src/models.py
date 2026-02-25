from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


def train_ols(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def train_rf(x_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model


def tune_rf(x_train, y_train):
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [0.5, 0.75, 1.0],
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
    )
    grid.fit(x_train, y_train)
    print(f"RF best params: {grid.best_params_}")
    return grid.best_estimator_


def train_xgb(x_train, y_train):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(x_train, y_train)
    return model


def tune_xgb(x_train, y_train):
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
    }
    grid = GridSearchCV(
        XGBRegressor(random_state=42),
        param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
    )
    grid.fit(x_train, y_train)
    print(f"XGB best params: {grid.best_params_}")
    return grid.best_estimator_
