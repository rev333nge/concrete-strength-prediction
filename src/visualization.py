import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocessing import load_data, cap_outliers, FEATURE_COLS, MODEL_FEATURE_COLS, TARGET_COL

FIGURES_DIR = "figures/"


def print_data_overview(df):
    print("=" * 55)
    print("PREGLED DATASETA")
    print("=" * 55)

    print(f"\nOblik dataseta: {df.shape} — {df.shape[0]} redova i {df.shape[1]} kolona\n")

    print("Informacije o kolonama:")
    print(df.info())

    print("\nOsnovne statistike:")
    print(df.describe().round(2))

    print("\nBroj nedostajućih vrednosti po koloni:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  Nema nedostajućih vrednosti.")
    else:
        print(missing[missing > 0])

    print("\nBroj duplikata:", df.duplicated().sum())
    print("=" * 55)


def plot_strength_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(df[TARGET_COL], bins=30, color="steelblue", edgecolor="white")
    ax.set_title("Distribucija čvrstoće betona")
    ax.set_xlabel("Čvrstoća (MPa)")
    ax.set_ylabel("Broj uzoraka")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "strength_distribution.png", dpi=150)
    plt.show()

def plot_boxplots(df):
    cols = FEATURE_COLS + [TARGET_COL]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].boxplot(df[col], patch_artist=True,
                        boxprops=dict(facecolor="steelblue", color="navy"),
                        medianprops=dict(color="white", linewidth=2))
        axes[i].set_title(col)
        axes[i].set_ylabel("Vrednost")

    plt.suptitle("Box plotovi — pre cappinga", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "boxplots_before.png", dpi=150)
    plt.show()


def plot_boxplots_after(df):
    cols = FEATURE_COLS + [TARGET_COL]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].boxplot(df[col], patch_artist=True,
                        boxprops=dict(facecolor="steelblue", color="navy"),
                        medianprops=dict(color="white", linewidth=2))
        axes[i].set_title(col)
        axes[i].set_ylabel("Vrednost")

    plt.suptitle("Box plotovi — nakon cappinga", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "boxplots_after.png", dpi=150)
    plt.show()


def plot_scatter_vs_strength(df):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLS):
        axes[i].scatter(df[col], df[TARGET_COL], alpha=0.4, color="steelblue", s=15)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Strength (MPa)")
        axes[i].set_title(f"{col} vs Strength")

    plt.suptitle("Odnos između sastojaka i čvrstoće betona", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "scatter_vs_strength.png", dpi=150)
    plt.show()


def plot_age_vs_strength(df):
    df = df.copy()
    bins = [0, 7, 14, 28, 56, 90, 180]
    labels = ["1–7", "8–14", "15–28", "29–56", "57–90", "91–180"]
    df["Age_group"] = pd.cut(df["Age"], bins=bins, labels=labels)

    groups = [df[df["Age_group"] == label][TARGET_COL].values for label in labels]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(groups, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", color="navy"),
                    medianprops=dict(color="white", linewidth=2))

    ax.set_title("Čvrstoća betona po starosnim grupama (dani)")
    ax.set_xlabel("Starost (dani)")
    ax.set_ylabel("Čvrstoća (MPa)")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "age_vs_strength.png", dpi=150)
    plt.show()

def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 8))

    corr = df[FEATURE_COLS + [TARGET_COL]].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
    )
    ax.set_title("Korelaciona matrica")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "correlation_heatmap.png", dpi=150)
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
        ax.set_xlabel("Stvarna čvrstoća (MPa)")
        ax.set_ylabel("Predviđena čvrstoća (MPa)")

    plt.suptitle("Stvarne vs predviđene vrednosti", fontsize=13)
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
        ax.set_xlabel("Predviđena čvrstoća (MPa)")
        ax.set_ylabel("Rezidual (MPa)")

    plt.suptitle("Reziduali po modelu", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "residuals.png", dpi=150)
    plt.show()


def plot_feature_importance(rf_model, xgb_model, ols_model):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # RF
    rf_imp = pd.Series(rf_model.feature_importances_, index=MODEL_FEATURE_COLS).sort_values()
    axes[0].barh(rf_imp.index, rf_imp.values, color="steelblue")
    axes[0].set_title("Random Forest")
    axes[0].set_xlabel("Važnost")

    # XGBoost
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=MODEL_FEATURE_COLS).sort_values()
    axes[1].barh(xgb_imp.index, xgb_imp.values, color="steelblue")
    axes[1].set_title("XGBoost")
    axes[1].set_xlabel("Važnost")

    # OLS koeficijenti
    ols_coef = pd.Series(np.abs(ols_model.coef_), index=MODEL_FEATURE_COLS).sort_values()
    axes[2].barh(ols_coef.index, ols_coef.values, color="steelblue")
    axes[2].set_title("OLS (aps. koeficijenti)")
    axes[2].set_xlabel("Apsolutna vrednost koeficijenta")

    plt.suptitle("Važnost feature-a po modelu", fontsize=13)
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
    df_capped = cap_outliers(df)
    plot_boxplots_after(df_capped)
