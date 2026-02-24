from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def train_ols(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def train_rf(x_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model


def train_xgb(x_train, y_train):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(x_train, y_train)
    return model
