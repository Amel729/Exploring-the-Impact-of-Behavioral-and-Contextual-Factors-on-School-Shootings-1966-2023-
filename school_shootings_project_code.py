"""
Exploring the Impact of Behavioral and Contextual Factors on School Shootings (1966-2023)
Python project script reconstructed from the project report.

What this script includes:
- data loading
- flexible column matching for slightly different CSV headers
- data cleaning and feature engineering
- exploratory plots
- correlation heatmap
- train/test split
- imputation + optional SMOTE
- Random Forest and Gradient Boosting models
- evaluation metrics and confusion matrices

Before running:
1) Put the CHDS School Shooting Safety Compendium CSV in the same folder
   OR update DATA_PATH below.
2) Install packages if needed:
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

Notes:
- Because dataset headers sometimes differ by export/version, this script uses
  helper functions that try multiple possible column names.
- This is an exploratory academic script, not a production risk scoring tool.
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_PATH = "school_shootings.csv"   # 
OUTPUT_DIR = "project_outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def normalize_name(name: str) -> str:
    """Normalize a column name for easier matching."""
    return (
        str(name)
        .strip()
        .lower()
        .replace("\n", " ")
        .replace("-", " ")
        .replace("/", " ")
        .replace(":", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("__", "_")
    )


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first matching column from a list of candidate names."""
    normalized_lookup = {normalize_name(col): col for col in df.columns}

    for candidate in candidates:
        c = normalize_name(candidate)
        if c in normalized_lookup:
            return normalized_lookup[c]

    # fallback: partial containment match
    for candidate in candidates:
        c = normalize_name(candidate)
        for norm_col, original_col in normalized_lookup.items():
            if c in norm_col or norm_col in c:
                return original_col
    return None


def to_numeric_series(df: pd.DataFrame, candidates: Iterable[str], fillna: Optional[float] = None) -> pd.Series:
    """Find a column and convert it to numeric."""
    col = find_column(df, candidates)
    if col is None:
        series = pd.Series(np.nan, index=df.index)
    else:
        series = pd.to_numeric(df[col], errors="coerce")

    if fillna is not None:
        series = series.fillna(fillna)
    return series


def to_string_series(df: pd.DataFrame, candidates: Iterable[str], fillna: str = "Unknown") -> pd.Series:
    """Find a column and convert it to string."""
    col = find_column(df, candidates)
    if col is None:
        return pd.Series(fillna, index=df.index)
    return df[col].astype(str).fillna(fillna)


def clean_binary_text(value: object) -> float:
    """Convert common yes/no text patterns to 1/0; otherwise NaN."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1", "present"}:
        return 1.0
    if text in {"no", "n", "false", "0", "absent"}:
        return 0.0
    return np.nan


def save_plot(filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
def load_data(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find '{data_path}'. Put your CSV in the same folder as this script "
            f"or update DATA_PATH at the top of the file."
        )
    df = pd.read_csv(data_path)
    print_section("RAW DATA OVERVIEW")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(list(df.columns))
    return df


# -----------------------------------------------------------------------------
# PREPARE DATA
# -----------------------------------------------------------------------------
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    # Year/date fields
    work["incident_year"] = to_numeric_series(
        work,
        [
            "year",
            "incident year",
            "school year",
            "date year",
        ],
    )

    date_col = find_column(work, ["date", "incident date", "event date"])
    if date_col is not None:
        parsed_dates = pd.to_datetime(work[date_col], errors="coerce")
        work["incident_date"] = parsed_dates
        work["incident_year"] = work["incident_year"].fillna(parsed_dates.dt.year)
        work["day_of_week"] = parsed_dates.dt.day_name()
    else:
        work["incident_date"] = pd.NaT
        work["day_of_week"] = to_string_series(work, ["day of week", "weekday"], fillna="Unknown")

    # Severity fields
    work["num_injured"] = to_numeric_series(
        work,
        ["injured", "num injured", "number injured", "injuries", "wounded"],
        fillna=0,
    )
    work["num_killed"] = to_numeric_series(
        work,
        ["killed", "num killed", "number killed", "fatalities", "deaths"],
        fillna=0,
    )
    work["total_casualties"] = work["num_injured"] + work["num_killed"]

    # Firearms field
    work["total_firearms"] = to_numeric_series(
        work,
        [
            "total firearms brought to the scene",
            "firearms brought",
            "number of firearms",
            "total firearms",
            "guns brought",
        ],
        fillna=0,
    )

    # Example behavioral/contextual indicators from the report
    text_binary_candidates = {
        "psychiatric_medication": [
            "psychiatric medication",
            "medication",
            "on psychiatric medication",
        ],
        "paranoia": ["paranoia", "paranoid"],
        "isolation": ["isolation", "social isolation", "isolated"],
        "depressed_mood": ["notably depressed mood", "depressed mood", "depression"],
        "childhood_trauma": ["childhood trauma", "history of trauma", "trauma"],
        "psychosis_motive": ["motive psychosis", "psychosis in the shooting", "psychosis"],
        "family_involvement": ["family member involvement", "family involvement", "family"],
        "daily_tasks": ["inability to perform daily tasks", "daily tasks impairment"],
        "school_performance": ["school performance", "academic performance"],
    }

    for new_col, candidate_names in text_binary_candidates.items():
        source_col = find_column(work, candidate_names)
        if source_col is None:
            work[new_col] = np.nan
            continue

        if pd.api.types.is_numeric_dtype(work[source_col]):
            work[new_col] = pd.to_numeric(work[source_col], errors="coerce")
        else:
            work[new_col] = work[source_col].map(clean_binary_text)

    # Location / context columns if available
    work["state"] = to_string_series(work, ["state", "incident state"], fillna="Unknown")
    work["school_level"] = to_string_series(work, ["school level", "grade level", "school type"], fillna="Unknown")

    # Drop rows where target is completely missing after derivation
    work = work.dropna(subset=["incident_year"], how="all").copy()

    # Build a severity class target from total casualties for classification
    # 0 = no casualties, 1 = low, 2 = medium/high
    work["severity_class"] = pd.cut(
        work["total_casualties"],
        bins=[-1, 0, 2, np.inf],
        labels=[0, 1, 2],
    ).astype("Int64")

    print_section("CLEANED DATA PREVIEW")
    print(work[[
        "incident_year", "num_injured", "num_killed", "total_casualties", "total_firearms",
        "day_of_week", "state", "school_level", "severity_class"
    ]].head())

    return work


# -----------------------------------------------------------------------------
# EXPLORATORY ANALYSIS
# -----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> None:
    print_section("DESCRIPTIVE STATISTICS")
    cols_to_describe = ["incident_year", "num_injured", "num_killed", "total_casualties", "total_firearms"]
    print(df[cols_to_describe].describe(include="all"))

    # Histogram: incidents over years
    plt.figure(figsize=(10, 6))
    sns.histplot(df["incident_year"].dropna(), bins=30, kde=True)
    plt.title("Distribution of School Shooting Incidents Over the Years")
    plt.xlabel("Incident Year")
    plt.ylabel("Frequency")
    save_plot("hist_incident_year.png")

    # Histogram: firearms brought
    plt.figure(figsize=(10, 6))
    sns.histplot(df["total_firearms"].dropna(), bins=30, kde=True)
    plt.title("Distribution of Total Firearms Brought to the Scene")
    plt.xlabel("Total Firearms")
    plt.ylabel("Frequency")
    save_plot("hist_total_firearms.png")

    # Boxplot by day of week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=df, x="day_of_week", y="total_firearms", order=day_order)
    plt.title("Total Firearms Brought to the Scene by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Total Firearms")
    plt.xticks(rotation=20)
    save_plot("boxplot_firearms_by_day.png")

    # Correlation heatmap
    corr_cols = [
        "num_injured",
        "num_killed",
        "total_casualties",
        "total_firearms",
        "psychiatric_medication",
        "paranoia",
        "isolation",
        "depressed_mood",
        "childhood_trauma",
        "psychosis_motive",
        "family_involvement",
        "daily_tasks",
        "school_performance",
    ]
    corr_df = df[corr_cols].copy()
    corr_df = corr_df.dropna(axis=1, how="all")

    if corr_df.shape[1] >= 2:
        plt.figure(figsize=(12, 9))
        sns.heatmap(corr_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        save_plot("correlation_heatmap.png")
    else:
        print("Not enough numeric columns available to create a correlation heatmap.")


# -----------------------------------------------------------------------------
# MODELING
# -----------------------------------------------------------------------------
def build_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = [
        "incident_year",
        "total_firearms",
        "day_of_week",
        "state",
        "school_level",
        "psychiatric_medication",
        "paranoia",
        "isolation",
        "depressed_mood",
        "childhood_trauma",
        "psychosis_motive",
        "family_involvement",
        "daily_tasks",
        "school_performance",
    ]

    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features].copy()
    y = df["severity_class"].astype(int)
    return X, y


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    preds = model.predict(X_test)

    print_section(f"{name} EVALUATION")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds, digits=4))

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    save_plot(f"confusion_matrix_{name.lower().replace(' ', '_')}.png")


def run_models(df: pd.DataFrame) -> None:
    X, y = build_feature_table(df)

    # Remove rows with missing target only
    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    print_section("TARGET DISTRIBUTION")
    print(y.value_counts(dropna=False).sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )

    # Random Forest / bagging-style model
    if IMBLEARN_AVAILABLE:
        rf_model = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    max_depth=None,
                    min_samples_leaf=2,
                ),
            ),
        ])
    else:
        rf_model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    max_depth=None,
                    min_samples_leaf=2,
                ),
            ),
        ])

    rf_model.fit(X_train, y_train)
    evaluate_model("Random Forest", rf_model, X_test, y_test)

    # Gradient Boosting
    gb_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    gb_model.fit(X_train, y_train)
    evaluate_model("Gradient Boosting", gb_model, X_test, y_test)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    df = load_data(DATA_PATH)
    clean_df = prepare_data(df)
    run_eda(clean_df)
    run_models(clean_df)

    print_section("DONE")
    print(f"All outputs were saved in: {os.path.abspath(OUTPUT_DIR)}")
    if not IMBLEARN_AVAILABLE:
        print(
            "Note: imbalanced-learn is not installed, so SMOTE was skipped. "
            "Install it with: pip install imbalanced-learn"
        )


if __name__ == "__main__":
    main()
