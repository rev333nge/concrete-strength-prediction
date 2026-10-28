import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocessing import load_data, cap_outliers, FEATURE_COLS, MODEL_FEATURE_COLS, TARGET_COL

FIGURES_DIR = "figures/"


def print_data_overview(df):
    print("=" * 55)
    print("DATASET OVERVIEW")
    print("=" * 55)

    print(f"\nDataset shape: {df.shape} — {df.shape[0]} rows and {df.shape[1]} columns\n")

    print("Column information:")
    print(df.info())

    print("\nSummary statistics:")
    print(df.describe().round(2))

    print("\nMissing values per column:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values.")
    else:
        print(missing[missing > 0])

    print("\nNumber of duplicates:", df.duplicated().sum())
    print("=" * 55)


def plot_strength_distribution(df):
    fig, ax = plt.subplots(figsize=(6.5, 4))

    ax.hist(df[TARGET_COL], bins=30, color="steelblue", edgecolor="white")
    ax.set_title("Concrete strength distribution")
    ax.set_xlabel("Strength (MPa)")
    ax.set_ylabel("Number of samples")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "strength_distribution.png", dpi=150)
    plt.show()

def plot_boxplots(df):
    cols = FEATURE_COLS + [TARGET_COL]

    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].boxplot(df[col], patch_artist=True,
                        boxprops=dict(facecolor="steelblue", color="navy"),
                        medianprops=dict(color="white", linewidth=2))
        axes[i].set_title(col)
        axes[i].set_ylabel("Value")

    plt.suptitle("Box plots before capping", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "boxplots_before.png", dpi=150)
    plt.show()


def plot_boxplots_after(df):
    cols = FEATURE_COLS + [TARGET_COL]

    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].boxplot(df[col], patch_artist=True,
                        boxprops=dict(facecolor="steelblue", color="navy"),
                        medianprops=dict(color="white", linewidth=2))
        axes[i].set_title(col)
        axes[i].set_ylabel("Value")

    plt.suptitle("Box plots after capping", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "boxplots_after.png", dpi=150)
    plt.show()


def plot_scatter_vs_strength(df):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLS):
        axes[i].scatter(df[col], df[TARGET_COL], alpha=0.4, color="steelblue", s=15)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Strength (MPa)")
        axes[i].set_title(f"{col} vs Strength")

    plt.suptitle("Relationship between ingredients and concrete strength", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "scatter_vs_strength.png", dpi=150)
    plt.show()


def plot_age_vs_strength(df):
    df = df.copy()
    bins = [0, 7, 14, 28, 56, 90, 180]
    labels = ["1–7", "8–14", "15–28", "29–56", "57–90", "91–180"]
    df["Age_group"] = pd.cut(df["Age"], bins=bins, labels=labels)

    groups = [df[df["Age_group"] == label][TARGET_COL].values for label in labels]

    fig, ax = plt.subplots(figsize=(9.5, 4))
    ax.boxplot(groups, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", color="navy"),
                    medianprops=dict(color="white", linewidth=2))

    ax.set_title("Concrete strength by age group (days)")
    ax.set_xlabel("Age (days)")
    ax.set_ylabel("Strength (MPa)")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "age_vs_strength.png", dpi=150)
    plt.show()

def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 6.5))

    corr = df[FEATURE_COLS + [TARGET_COL]].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
    )
    ax.set_title("Correlation matrix")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "correlation_heatmap.png", dpi=150)
    plt.show()

def plot_single_model_diagnostics(name, model, x_test, y_test):
    y_pred = model.predict(x_test)
    residuals = y_pred - y_test

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].scatter(y_test, y_pred, alpha=0.4, color="steelblue", s=15)
    axes[0].plot(lims, lims, color="red", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Actual strength (MPa)")
    axes[0].set_ylabel("Predicted strength (MPa)")
    axes[0].set_title("Actual vs predicted values")

    axes[1].scatter(y_pred, residuals, alpha=0.4, color="steelblue", s=15)
    axes[1].axhline(0, color="red", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Predicted strength (MPa)")
    axes[1].set_ylabel("Residual (MPa)")
    axes[1].set_title("Residuals")

    plt.suptitle(name, fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + f"diagnostics_{name.lower()}.png", dpi=150)
    plt.show()


def plot_residual_histogram(model, x, y):
    y_pred = model.predict(x)
    residuals = y_pred - y

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=25, color="steelblue", edgecolor="white", alpha=0.7)
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Residual (MPa)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual distribution (OLS)")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "residual_histogram_ols.png", dpi=150)
    plt.show()


def plot_feature_importance_single(model, name):
    imp = pd.Series(model.feature_importances_, index=model.feature_names_in_).sort_values()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color="steelblue")
    ax.set_title(f"Feature importance - {name}")
    ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR + f"feature_importance_{name.lower()}.png", dpi=150)
    plt.show()


def plot_ablation(ablation_results):
    steps = [r["n_features"] for r in ablation_results]
    val_rmse = [r["val_rmse"] for r in ablation_results]
    train_rmse = [r["train_rmse"] for r in ablation_results]

    best_idx = int(np.argmin(val_rmse))
    best_n = steps[best_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, train_rmse, "s--", color="coral", alpha=0.7, label="Train RMSE")
    ax.plot(steps, val_rmse, "o-", color="steelblue", label="Validation RMSE")

    # Annotate removed features
    for r in ablation_results:
        if r["removed"] is not None:
            ax.annotate(
                f"−{r['removed']}",
                xy=(r["n_features"], r["val_rmse"]),
                xytext=(0, 12),
                textcoords="offset points",
                fontsize=7,
                ha="center",
                rotation=30,
            )

    # Mark optimal
    ax.axvline(best_n, color="green", linestyle=":", alpha=0.7)
    ax.annotate(
        f"Optimal: {best_n} feat.",
        xy=(best_n, val_rmse[best_idx]),
        xytext=(15, -20),
        textcoords="offset points",
        fontsize=9,
        color="green",
        arrowprops=dict(arrowstyle="->", color="green"),
    )

    ax.set_xlabel("Number of features")
    ax.set_ylabel("RMSE (MPa)")
    ax.set_title("OLS ablation analysis - backward elimination")
    ax.legend()
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "ols_ablation.png", dpi=150)
    plt.show()


def plot_actual_vs_predicted(models_dict, splits_dict):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (name, model) in zip(axes, models_dict.items()):
        x_test, y_test = splits_dict[name]
        y_pred = model.predict(x_test)

        ax.scatter(y_test, y_pred, alpha=0.4, color="steelblue", s=15)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, color="red", linewidth=1.5, linestyle="--")
        ax.set_title(name)
        ax.set_xlabel("Actual strength (MPa)")
        ax.set_ylabel("Predicted strength (MPa)")

    plt.suptitle("Actual vs predicted values", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "actual_vs_predicted.png", dpi=150)
    plt.show()


def plot_residuals(models_dict, splits_dict):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (name, model) in zip(axes, models_dict.items()):
        x_test, y_test = splits_dict[name]
        y_pred = model.predict(x_test)
        residuals = y_pred - y_test

        ax.scatter(y_pred, residuals, alpha=0.4, color="steelblue", s=15)
        ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
        ax.set_title(name)
        ax.set_xlabel("Predicted strength (MPa)")
        ax.set_ylabel("Residual (MPa)")

    plt.suptitle("Residuals by model", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "residuals.png", dpi=150)
    plt.show()


def plot_feature_importance(rf_model, xgb_model, ols_model):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # RF
    rf_imp = pd.Series(rf_model.feature_importances_, index=MODEL_FEATURE_COLS).sort_values()
    axes[0].barh(rf_imp.index, rf_imp.values, color="steelblue")
    axes[0].set_title("Random Forest")
    axes[0].set_xlabel("Importance")

    # XGBoost
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=MODEL_FEATURE_COLS).sort_values()
    axes[1].barh(xgb_imp.index, xgb_imp.values, color="steelblue")
    axes[1].set_title("XGBoost")
    axes[1].set_xlabel("Importance")

    # OLS coefficients
    ols_features = getattr(ols_model, "feature_names_in_", MODEL_FEATURE_COLS)
    ols_coef = pd.Series(np.abs(ols_model.coef_), index=ols_features).sort_values()
    axes[2].barh(ols_coef.index, ols_coef.values, color="steelblue")
    axes[2].set_title("OLS (abs. coefficients)")
    axes[2].set_xlabel("Absolute coefficient value")

    plt.suptitle("Feature importance by model", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "feature_importance.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    df = load_data()
    print_data_overview(df)
    plot_strength_distribution(df)
    plot_scatter_vs_strength(df)
    plot_age_vs_strength(df)
    plot_correlation_heatmap(df)
    plot_boxplots(df)
    df_capped, _ = cap_outliers(df)
    plot_boxplots_after(df_capped)
