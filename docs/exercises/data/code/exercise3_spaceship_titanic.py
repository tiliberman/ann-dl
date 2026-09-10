"""
Exercise 3 — Preparing real-world data for a tanh network.

Takes the Kaggle Spaceship Titanic training file and turns it into a feature matrix a
network with tanh hidden units can actually be fed: no missing values, categorical
features encoded, heavy tails compressed, everything inside [-1, 1].

Every statistic used by a transformation -- median, category list, min/max -- is fitted
on the training split only and then applied to the test split. That ordering is the
whole point of item B: fitting on the full table before splitting leaks test
information into the transformation and inflates the reported performance.

Run from the repository root:
    python docs/exercises/data/code/exercise3_spaceship_titanic.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

SEED = 42

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data" / "train.csv"
FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

SPENDING = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
DROP = ["PassengerId", "Cabin", "Name"]
TARGET = "Transported"


def load():
    """Read train.csv -- the only labelled file in the competition."""
    if not DATA.exists():
        raise SystemExit(
            f"Missing {DATA}.\nDownload train.csv from "
            "https://www.kaggle.com/competitions/spaceship-titanic/data "
            f"and place it at {DATA}."
        )
    return pd.read_csv(DATA)


def describe(df):
    """Item A — target, feature types, missing values and the shape of the spending columns."""
    print("=" * 74)
    print("A - GETTING TO KNOW THE DATA")
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

    balance = df[TARGET].value_counts(normalize=True).sort_index()
    counts = df[TARGET].value_counts().sort_index()
    print(f"\nTarget '{TARGET}' -- was the passenger transported to another dimension?")
    for value in balance.index:
        print(f"  {str(value):<6} {counts[value]:>5} rows  ({balance[value]:.4%})")
    print(f"  the dataset is essentially balanced: "
          f"{balance[True]:.2%} positive against {balance[False]:.2%} negative")

    numerical = df.select_dtypes(include=np.number).columns.tolist()
    other = [c for c in df.columns if c not in numerical and c != TARGET]
    print(f"\nNumerical features ({len(numerical)}): {numerical}")
    print(f"Categorical / identifier features ({len(other)}): {other}")

    print("\nMissing values per column")
    missing = pd.DataFrame({
        "missing": df.isna().sum(),
        "percent": (df.isna().mean() * 100).round(2),
    }).sort_values("missing", ascending=False)
    print(missing.to_string())
    print(f"\n  rows with at least one missing value: {df.isna().any(axis=1).sum()} "
          f"({df.isna().any(axis=1).mean():.2%})")

    print("\nSpending columns -- mean vs. median vs. maximum")
    stats = df[SPENDING].agg(["mean", "median", "max"]).T
    stats["mean/median"] = np.where(stats["median"] > 0,
                                    stats["mean"] / stats["median"], np.inf)
    stats["zeros %"] = (df[SPENDING] == 0).mean().values * 100
    print(stats.round(2).to_string())
    print("\n  The median is 0 for every spending column while the means run into the\n"
          "  hundreds and the maxima into the tens of thousands: most passengers spend\n"
          "  nothing at all and a small minority spends enormously. These are heavy\n"
          "  right-tailed distributions, not symmetric ones.")

    return missing


def split(df):
    """Item B — stratified 80/20 split, done before any statistic is computed."""
    features = df.drop(columns=DROP + [TARGET])
    target = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=SEED, stratify=target
    )
    print("\n" + "=" * 74)
    print("B - SPLIT BEFORE TRANSFORMING")
    print(f"\nDropped as identifiers / free text: {DROP}")
    print(f"  train: {X_train.shape[0]} rows  positive class {y_train.mean():.4%}")
    print(f"  test : {X_test.shape[0]} rows  positive class {y_test.mean():.4%}")
    print("  stratify=y keeps the class balance identical in both splits")
    return X_train, X_test, y_train, y_test


def preprocess(X_train, X_test):
    """Item C — impute, engineer, compress tails, encode, scale. Fit on train only."""
    print("\n" + "=" * 74)
    print("C - PREPROCESSING")

    numeric_cols = ["Age"] + SPENDING

    # --- 1. Missing values ------------------------------------------------------
    # Numerical: median. The spending columns are heavily right-skewed, so the mean is
    # dragged upwards by a few big spenders; the median is not. It also happens to be 0
    # for the spending columns, which matches the dominant behaviour (most passengers
    # spend nothing) instead of inventing a purchase.
    # Categorical: most frequent. There is no meaningful "average" of a home planet, and
    # the missing share is small (~2%), so the mode adds little distortion.
    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    train = X_train.copy()
    test = X_test.copy()
    train[numeric_cols] = numeric_imputer.fit_transform(train[numeric_cols])
    test[numeric_cols] = numeric_imputer.transform(test[numeric_cols])
    train[CATEGORICAL] = categorical_imputer.fit_transform(train[CATEGORICAL])
    test[CATEGORICAL] = categorical_imputer.transform(test[CATEGORICAL])

    print("\n1. Imputation (fitted on train, applied to test)")
    for col, value in zip(numeric_cols, numeric_imputer.statistics_):
        print(f"   {col:<14} median from train -> {value}")
    for col, value in zip(CATEGORICAL, categorical_imputer.statistics_):
        print(f"   {col:<14} mode   from train -> {value}")

    # --- 2. Feature engineering -------------------------------------------------
    # Built from the raw amounts, before the log, so it is a real total in currency.
    train["TotalSpend"] = train[SPENDING].sum(axis=1)
    test["TotalSpend"] = test[SPENDING].sum(axis=1)
    print("\n2. Feature engineering: TotalSpend = sum of the five spending columns")
    print(f"   train TotalSpend  mean {train['TotalSpend'].mean():.2f}  "
          f"median {train['TotalSpend'].median():.2f}  max {train['TotalSpend'].max():.2f}")

    # --- 3. Heavy tails ---------------------------------------------------------
    # log(1+x) keeps the zeros at zero and pulls a 20000-credit outlier to ~10, so the
    # spread of the column stops being dictated by a handful of passengers.
    spend_log = SPENDING + ["TotalSpend"]
    raw_foodcourt_train = train["FoodCourt"].copy()
    train[spend_log] = np.log1p(train[spend_log])
    test[spend_log] = np.log1p(test[spend_log])
    print("\n3. log(1 + x) applied to the spending columns and to TotalSpend")
    print(f"   FoodCourt before: max {raw_foodcourt_train.max():.1f}, "
          f"skew {raw_foodcourt_train.skew():.2f}")
    print(f"   FoodCourt after : max {train['FoodCourt'].max():.3f}, "
          f"skew {train['FoodCourt'].skew():.2f}")

    # --- 4. Categorical encoding ------------------------------------------------
    # handle_unknown="ignore": a category present only in the test set is encoded as all
    # zeros across that feature's dummy columns instead of raising. The column count is
    # therefore fixed by the training set, which is what keeps the matrix feedable to a
    # network whose input layer has a fixed width.
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_dummies = encoder.fit_transform(train[CATEGORICAL])
    test_dummies = encoder.transform(test[CATEGORICAL])
    dummy_names = encoder.get_feature_names_out(CATEGORICAL)

    print("\n4. One-hot encoding (categories learned from train only)")
    for col, cats in zip(CATEGORICAL, encoder.categories_):
        print(f"   {col:<14} -> {[str(c) for c in cats]}")
    unseen = {
        col: sorted(set(test[col].unique()) - set(cats))
        for col, cats in zip(CATEGORICAL, encoder.categories_)
    }
    unseen = {k: v for k, v in unseen.items() if v}
    print(f"   categories in test but not in train: {unseen if unseen else 'none'}")
    print("   any such category would become an all-zero row, never an error")

    # --- 5. Scaling -------------------------------------------------------------
    # Normalisation to [-1, 1] rather than standardisation: tanh saturates outside
    # roughly [-2, 2], and standardisation leaves the log-spending columns with values
    # past |z| = 3, which land in the flat part of the curve where the gradient dies.
    # Min-max to [-1, 1] puts every training value inside the responsive region by
    # construction, and centres it on 0, which is where tanh has its steepest slope.
    numeric_final = ["Age"] + SPENDING + ["TotalSpend"]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_numeric = scaler.fit_transform(train[numeric_final])
    test_numeric = scaler.transform(test[numeric_final])

    print("\n5. Scaling: MinMaxScaler to [-1, 1], fitted on train")
    print(f"   train numeric range: [{train_numeric.min():.4f}, {train_numeric.max():.4f}]")
    print(f"   test  numeric range: [{test_numeric.min():.4f}, {test_numeric.max():.4f}]")
    print("   the test range may fall slightly outside [-1, 1]: the scaler only ever saw\n"
          "   the training minimum and maximum, and that is exactly the point")

    feature_names = numeric_final + list(dummy_names)
    train_matrix = pd.DataFrame(np.hstack([train_numeric, train_dummies]),
                                columns=feature_names, index=train.index)
    test_matrix = pd.DataFrame(np.hstack([test_numeric, test_dummies]),
                               columns=feature_names, index=test.index)

    return train_matrix, test_matrix, raw_foodcourt_train


def figure6(raw_foodcourt, processed_foodcourt):
    """Figure 6 — FoodCourt before and after the log + scaling pipeline."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(raw_foodcourt, bins=60, color="#d62728", alpha=0.8)
    axes[0].set_title(f"Before — raw FoodCourt (skew {raw_foodcourt.skew():.2f})")
    axes[0].set_xlabel("FoodCourt (credits)")
    axes[0].set_ylabel("Number of passengers")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)

    axes[1].hist(processed_foodcourt, bins=60, color="#2ca02c", alpha=0.8)
    axes[1].set_title(f"After — log(1+x) then scaled to [-1, 1] "
                      f"(skew {processed_foodcourt.skew():.2f})")
    axes[1].set_xlabel("FoodCourt (transformed)")
    axes[1].set_ylabel("Number of passengers")
    axes[1].grid(alpha=0.25)

    fig.suptitle("Figure 6 — Effect of the log transform and scaling on a heavy-tailed "
                 "feature (training set)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06-foodcourt-before-after.png", dpi=150)
    plt.close(fig)


def verify(train_matrix, test_matrix, y_train, y_test):
    """Item D — the checks that decide whether this matrix can be fed to the network."""
    print("\n" + "=" * 74)
    print("D - FINAL CHECKS")
    print(f"\n  NaN remaining in train: {int(train_matrix.isna().sum().sum())}")
    print(f"  NaN remaining in test : {int(test_matrix.isna().sum().sum())}")
    print(f"  train feature matrix shape: {train_matrix.shape}   labels: {y_train.shape}")
    print(f"  test  feature matrix shape: {test_matrix.shape}   labels: {y_test.shape}")
    print(f"  train value range: [{train_matrix.values.min():.4f}, "
          f"{train_matrix.values.max():.4f}]")
    print(f"  test  value range: [{test_matrix.values.min():.4f}, "
          f"{test_matrix.values.max():.4f}]")
    within = np.abs(test_matrix.values).max() <= 1.0
    print(f"  every test value inside [-1, 1]: {within}")
    print("\n  Feature columns:")
    for name in train_matrix.columns:
        print(f"    {name}")


def main():
    df = load()
    describe(df)
    X_train, X_test, y_train, y_test = split(df)
    train_matrix, test_matrix, raw_foodcourt = preprocess(X_train, X_test)

    print("\n  FoodCourt on the TRAINING set, before transforming: "
          f"mean {raw_foodcourt.mean():.2f}, median {raw_foodcourt.median():.2f}")

    figure6(raw_foodcourt, train_matrix["FoodCourt"])
    verify(train_matrix, test_matrix, y_train, y_test)

    print("\nFigures written to", FIGURES)
    print("=" * 74)


if __name__ == "__main__":
    main()
