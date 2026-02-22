import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_data, FEATURE_COLS, TARGET_COL

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

    plt.suptitle("Box plotovi — pregled outliera po promenljivoj", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR + "boxplots.png", dpi=150)
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


if __name__ == "__main__":
    df = load_data()
    print_data_overview(df)
    plot_strength_distribution(df)
    plot_boxplots(df)
    plot_correlation_heatmap(df)
