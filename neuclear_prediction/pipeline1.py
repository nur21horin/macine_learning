"""
FULL RESEARCH PIPELINE
===============================================================

Imbalance-Aware Machine Learning for Anomaly Detection
in Nuclear Power Plant Cyber-Physical System (CPS) Data

IMPORTANT FIX:
-------------
The original pipeline created y BEFORE sorting the dataframe by
Timestamp. This could misalign features and labels.

This version:
    1. Loads the dataset
    2. Converts Timestamp
    3. Sorts the complete dataframe by Timestamp
    4. Creates y AFTER sorting
    5. Builds raw / temporal / domain-inspired features
    6. Runs data-quality diagnostics
    7. Runs imbalance-aware supervised ML experiments
    8. Uses a chronological train/test split
    9. Applies SMOTE/ADASYN ONLY to training data
   10. Evaluates Recall, Precision, F1, PR-AUC, ROC-AUC, MCC
   11. Performs SHAP stability analysis
   12. Performs shuffled-label control
   13. Performs anomaly-case error analysis

Required packages:
    pandas
    numpy
    scikit-learn
    imbalanced-learn
    xgboost
    shap
    scipy

Install:
    python -m pip install pandas numpy scikit-learn imbalanced-learn xgboost shap scipy

Run:
    python pipiline.py
"""

# ===============================================================
# IMPORTS
# ===============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    brier_score_loss,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix
)

from sklearn.feature_selection import mutual_info_classif

from imblearn.over_sampling import SMOTE, ADASYN

from xgboost import XGBClassifier

from scipy.stats import norm

import shap


# ===============================================================
# CONFIGURATION
# ===============================================================

RANDOM_STATE = 42

ROOT_DIR = Path(__file__).resolve().parent

DATA_PATH = ROOT_DIR / "Nuclear_Power_Plant_CPS_Dataset.csv"

OUT_DIR = ROOT_DIR / "outputs"

# Bootstrap iterations
# Increase to 1000 or 2000 for final publication experiment
N_BOOTSTRAP = 200

# SHAP CV folds
N_CV_FOLDS = 5

# Number of shuffled-label experiments
# Increase to 20 for final paper
N_SHUFFLES = 8

# Number of top SHAP features
TOP_K = 5

# Test proportion
TEST_SIZE = 0.15


# Models used in the supervised study
SUPERVISED_MODELS = [
    "LogisticRegression",
    "RandomForest",
    "XGBoost"
]


# ===============================================================
# HELPER
# ===============================================================

def safe_numeric_conversion(df):
    """
    Convert all non-target sensor columns to numeric when possible.
    """

    df = df.copy()

    for col in df.columns:

        if col in ["Timestamp", "Anomaly Detected"]:
            continue

        # Pump Status may contain ON/OFF
        if col == "Pump Status":

            if df[col].dtype == object:

                mapping = {
                    "ON": 1,
                    "OFF": 0,
                    "on": 1,
                    "off": 0,
                    "1": 1,
                    "0": 0
                }

                df[col] = df[col].map(mapping)

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        else:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


# ===============================================================
# SECTION A
# DATA LOADING + FEATURE ENGINEERING
# ===============================================================

def load_and_prepare_dataset():

    print("\n======================================================")
    print("SECTION A: DATA LOADING AND PREPARATION")
    print("======================================================")

    print(f"Dataset path: {DATA_PATH}")

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_PATH}\n\n"
            f"Make sure Nuclear_Power_Plant_CPS_Dataset.csv "
            f"is inside:\n{ROOT_DIR}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Original shape: {df.shape}")

    required_columns = [
        "Timestamp",
        "Anomaly Detected"
    ]

    for col in required_columns:

        if col not in df.columns:

            raise ValueError(
                f"Required column '{col}' not found."
            )

    # -----------------------------------------------------------
    # Timestamp
    # -----------------------------------------------------------

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    # Remove invalid timestamps
    df = df.dropna(
        subset=["Timestamp"]
    ).copy()

    # -----------------------------------------------------------
    # Convert sensor columns
    # -----------------------------------------------------------

    df = safe_numeric_conversion(df)

    # -----------------------------------------------------------
    # Sort COMPLETE dataframe
    # -----------------------------------------------------------

    df = df.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    # -----------------------------------------------------------
    # Handle missing numeric values
    # -----------------------------------------------------------

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    numeric_sensor_cols = [
        c for c in numeric_cols
        if c != "Anomaly Detected"
    ]

    for col in numeric_sensor_cols:

        if df[col].isna().sum() > 0:

            df[col] = df[col].interpolate(
                method="linear"
            )

            df[col] = df[col].bfill()
            df[col] = df[col].ffill()

    # -----------------------------------------------------------
    # CREATE TARGET AFTER SORTING
    # -----------------------------------------------------------

    y = (
        df["Anomaly Detected"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
        .astype(int)
        .values
    )

    print("\nDataset after preparation:")
    print(f"Samples: {len(df)}")

    print("\nTarget distribution:")

    class_counts = pd.Series(y).value_counts().sort_index()

    for cls, count in class_counts.items():

        label = "Normal" if cls == 0 else "Anomaly"

        percentage = (
            count / len(y) * 100
        )

        print(
            f"  {label}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    return df, y


# ===============================================================
# FEATURE ENGINEERING
# ===============================================================

def build_feature_sets(df):

    """
    Build three feature representations:

        1. raw
        2. temporal
        3. domain-inspired

    IMPORTANT:
    df is already sorted by Timestamp.
    """

    base = df.copy()

    # -----------------------------------------------------------
    # Raw columns
    # -----------------------------------------------------------

    raw_cols = [
        c for c in base.columns
        if c not in [
            "Timestamp",
            "Anomaly Detected"
        ]
    ]

    # Ensure Pump Status is numeric
    if "Pump Status" in base.columns:

        base["Pump Status"] = pd.to_numeric(
            base["Pump Status"],
            errors="coerce"
        ).fillna(0)

    sets = {}

    # ===========================================================
    # RAW FEATURES
    # ===========================================================

    raw = base[raw_cols].copy()

    sets["raw"] = (
        raw,
        list(raw.columns)
    )

    # ===========================================================
    # TEMPORAL FEATURES
    # ===========================================================

    temporal = base[raw_cols].copy()

    sensor_cols = [
        c for c in raw_cols
        if c != "Pump Status"
    ]

    # Rolling features
    for window in [5, 10, 15]:

        for col in sensor_cols:

            temporal[
                f"{col}_roll{window}_mean"
            ] = (
                base[col]
                .rolling(
                    window=window,
                    min_periods=1
                )
                .mean()
            )

            temporal[
                f"{col}_roll{window}_std"
            ] = (
                base[col]
                .rolling(
                    window=window,
                    min_periods=1
                )
                .std()
                .fillna(0)
            )

    # First differences
    for col in sensor_cols:

        temporal[
            f"{col}_delta1"
        ] = (
            base[col]
            .diff()
            .fillna(0)
        )

    sets["temporal"] = (
        temporal,
        list(temporal.columns)
    )

    # ===========================================================
    # DOMAIN-INSPIRED FEATURES
    # ===========================================================

    physics = base[raw_cols].copy()

    # -----------------------------------------------------------
    # Power residual
    #
    # NOTE:
    # This is a DOMAIN-INSPIRED composite feature.
    # It is NOT a calibrated nuclear engineering equation.
    # -----------------------------------------------------------

    expected_power = (
        0.5 * base["Steam Flow Rate (kg/s)"]
        +
        0.01 * base["Turbine Speed (RPM)"]
    )

    physics["power_residual"] = (
        base["Power Output (MW)"]
        -
        expected_power
    )

    # -----------------------------------------------------------
    # Pressure-temperature residual
    # -----------------------------------------------------------

    expected_pressure = (
        0.02 *
        base["Reactor Temp (°C)"]
    )

    physics["pressure_temp_residual"] = (
        base["Pressure (MPa)"]
        -
        expected_pressure
    )

    # -----------------------------------------------------------
    # Coolant per power
    # -----------------------------------------------------------

    power_safe = (
        base["Power Output (MW)"]
        .replace(0, np.nan)
    )

    physics["coolant_per_power"] = (
        base["Coolant Flow Rate (L/s)"]
        /
        power_safe
    )

    physics["coolant_per_power"] = (
        physics["coolant_per_power"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    sets["physics"] = (
        physics,
        list(physics.columns)
    )

    return sets, base


# ===============================================================
# SECTION B
# DATA QUALITY DIAGNOSTICS
# ===============================================================

def run_diagnostics(X, y, feature_cols):

    rows = []

    for col in feature_cols:

        values = X[col].values

        try:

            corr = np.corrcoef(
                values,
                y
            )[0, 1]

        except Exception:

            corr = np.nan

        rows.append({
            "feature": col,
            "pearson_corr": corr
        })

    corr_df = pd.DataFrame(rows)

    corr_df = corr_df.sort_values(
        "pearson_corr",
        key=lambda x: abs(x),
        ascending=False
    )

    # -----------------------------------------------------------
    # Mutual Information
    # -----------------------------------------------------------

    X_clean = X[feature_cols].copy()

    X_clean = X_clean.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_clean = X_clean.fillna(
        X_clean.median()
    )

    mi = mutual_info_classif(
        X_clean,
        y,
        random_state=RANDOM_STATE
    )

    mi_df = pd.DataFrame({
        "feature": feature_cols,
        "mutual_info": mi
    })

    mi_df = mi_df.sort_values(
        "mutual_info",
        ascending=False
    )

    diagnostics = corr_df.merge(
        mi_df,
        on="feature"
    )

    return diagnostics


# ===============================================================
# SECTION C
# IMBALANCE HANDLING
# ===============================================================

def apply_regime(
    X_train,
    y_train,
    regime,
    random_state=RANDOM_STATE
):

    # -----------------------------------------------------------
    # No balancing
    # -----------------------------------------------------------

    if regime == "none":

        return (
            X_train,
            y_train
        )

    # -----------------------------------------------------------
    # Class weighting
    # -----------------------------------------------------------

    if regime == "class_weight":

        return (
            X_train,
            y_train
        )

    # -----------------------------------------------------------
    # SMOTE
    # -----------------------------------------------------------

    if regime == "smote":

        minority_count = (
            y_train == 1
        ).sum()

        k = min(
            5,
            max(
                1,
                minority_count - 1
            )
        )

        smote = SMOTE(
            random_state=random_state,
            k_neighbors=k
        )

        return smote.fit_resample(
            X_train,
            y_train
        )

    # -----------------------------------------------------------
    # ADASYN
    # -----------------------------------------------------------

    if regime == "adasyn":

        minority_count = (
            y_train == 1
        ).sum()

        k = min(
            5,
            max(
                1,
                minority_count - 1
            )
        )

        adasyn = ADASYN(
            random_state=random_state,
            n_neighbors=k
        )

        return adasyn.fit_resample(
            X_train,
            y_train
        )

    raise ValueError(
        f"Unknown imbalance regime: {regime}"
    )


# ===============================================================
# MODELS
# ===============================================================

def get_models(
    class_weight=None,
    scale_pos_weight=None
):

    return {

        "LogisticRegression":

            LogisticRegression(
                max_iter=2000,
                class_weight=class_weight,
                random_state=RANDOM_STATE
            ),

        "RandomForest":

            RandomForestClassifier(
                n_estimators=300,
                class_weight=class_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "XGBoost":

            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
                scale_pos_weight=(
                    scale_pos_weight
                    if scale_pos_weight
                    else 1
                )
            )
    }


# ===============================================================
# CHRONOLOGICAL SPLIT
# ===============================================================

def chronological_split(
    X,
    y,
    test_size=TEST_SIZE
):

    n = len(X)

    test_start = int(
        n * (1 - test_size)
    )

    X_train = X.iloc[
        :test_start
    ].copy()

    X_test = X.iloc[
        test_start:
    ].copy()

    y_train = y[
        :test_start
    ]

    y_test = y[
        test_start:
    ]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ===============================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ===============================================================

def bootstrap_ci(
    y_true,
    y_score,
    metric_fn,
    n_boot=N_BOOTSTRAP,
    seed=RANDOM_STATE
):

    rng = np.random.RandomState(
        seed
    )

    idx_pos = np.where(
        y_true == 1
    )[0]

    idx_neg = np.where(
        y_true == 0
    )[0]

    stats = []

    if len(idx_pos) == 0 or len(idx_neg) == 0:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    for _ in range(n_boot):

        bootstrap_idx = np.concatenate([

            rng.choice(
                idx_pos,
                size=len(idx_pos),
                replace=True
            ),

            rng.choice(
                idx_neg,
                size=len(idx_neg),
                replace=True
            )

        ])

        try:

            value = metric_fn(
                y_true[bootstrap_idx],
                y_score[bootstrap_idx]
            )

            stats.append(value)

        except Exception:

            continue

    if len(stats) == 0:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    stats = np.array(stats)

    lo, hi = np.percentile(
        stats,
        [2.5, 97.5]
    )

    return (
        float(np.mean(stats)),
        float(lo),
        float(hi)
    )


# ===============================================================
# MODEL SCORES
# ===============================================================

def get_probability_scores(
    model,
    X_test
):

    return model.predict_proba(
        X_test
    )[:, 1]


# ===============================================================
# MODEL EVALUATION
# ===============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    feature_set,
    regime,
    model_name
):

    scores = get_probability_scores(
        model,
        X_test
    )

    # Default classification threshold
    predictions = (
        scores >= 0.5
    ).astype(int)

    roc = roc_auc_score(
        y_test,
        scores
    )

    pr_auc = average_precision_score(
        y_test,
        scores
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        predictions
    )

    balanced_acc = balanced_accuracy_score(
        y_test,
        predictions
    )

    brier = brier_score_loss(
        y_test,
        np.clip(scores, 0, 1)
    )

    roc_mean, roc_lo, roc_hi = bootstrap_ci(
        y_test,
        scores,
        roc_auc_score
    )

    pr_mean, pr_lo, pr_hi = bootstrap_ci(
        y_test,
        scores,
        average_precision_score
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {

        "feature_set": feature_set,

        "regime": regime,

        "model": model_name,

        "ROC_AUC": roc,

        "ROC_AUC_CI_low": roc_lo,

        "ROC_AUC_CI_high": roc_hi,

        "PR_AUC": pr_auc,

        "PR_AUC_CI_low": pr_lo,

        "PR_AUC_CI_high": pr_hi,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "MCC": mcc,

        "Balanced_Accuracy": balanced_acc,

        "Brier": brier,

        "TN": tn,

        "FP": fp,

        "FN": fn,

        "TP": tp
    }


# ===============================================================
# MODEL GRID
# ===============================================================

def run_grid_for_feature_set(
    X,
    y,
    feature_set_name,
    feature_cols,
    regimes=None,
    model_names=None
):

    if regimes is None:

        regimes = [
            "none",
            "class_weight",
            "smote",
            "adasyn"
        ]

    if model_names is None:

        model_names = SUPERVISED_MODELS

    # -----------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # -----------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = chronological_split(
        X,
        y
    )

    print(
        f"    Train samples: {len(X_train)}"
    )

    print(
        f"    Test samples: {len(X_test)}"
    )

    print(
        f"    Train anomalies: {y_train.sum()}"
    )

    print(
        f"    Test anomalies: {y_test.sum()}"
    )

    # -----------------------------------------------------------
    # Scaling
    # -----------------------------------------------------------

    scaler = RobustScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )

    results = []

    # -----------------------------------------------------------
    # Run regimes
    # -----------------------------------------------------------

    for regime in regimes:

        print(
            f"      Regime: {regime}"
        )

        # Class weight
        if regime == "class_weight":

            X_resampled = (
                X_train_scaled
            )

            y_resampled = y_train

            class_weight = "balanced"

            scale_pos_weight = (
                (y_train == 0).sum()
                /
                max(
                    (y_train == 1).sum(),
                    1
                )
            )

        else:

            X_resampled, y_resampled = (
                apply_regime(
                    X_train_scaled,
                    y_train,
                    regime
                )
            )

            class_weight = None
            scale_pos_weight = None

        # -------------------------------------------------------
        # Models
        # -------------------------------------------------------

        models = get_models(
            class_weight=class_weight,
            scale_pos_weight=scale_pos_weight
        )

        models = {
            name: model
            for name, model in models.items()
            if name in model_names
        }

        for model_name, model in models.items():

            print(
                f"        Training {model_name}..."
            )

            try:

                model.fit(
                    X_resampled,
                    y_resampled
                )

                result = evaluate_model(
                    model=model,
                    X_test=X_test_scaled,
                    y_test=y_test,
                    feature_set=feature_set_name,
                    regime=regime,
                    model_name=model_name
                )

                results.append(result)

            except Exception as error:

                print(
                    f"        ERROR: {error}"
                )

                results.append({

                    "feature_set":
                        feature_set_name,

                    "regime":
                        regime,

                    "model":
                        model_name,

                    "ROC_AUC":
                        np.nan,

                    "ROC_AUC_CI_low":
                        np.nan,

                    "ROC_AUC_CI_high":
                        np.nan,

                    "PR_AUC":
                        np.nan,

                    "PR_AUC_CI_low":
                        np.nan,

                    "PR_AUC_CI_high":
                        np.nan,

                    "Precision":
                        np.nan,

                    "Recall":
                        np.nan,

                    "F1":
                        np.nan,

                    "MCC":
                        np.nan,

                    "Balanced_Accuracy":
                        np.nan,

                    "Brier":
                        np.nan,

                    "TN":
                        np.nan,

                    "FP":
                        np.nan,

                    "FN":
                        np.nan,

                    "TP":
                        np.nan,

                    "error":
                        str(error)
                })

    return (
        pd.DataFrame(results),
        (
            X_test_scaled,
            y_test
        )
    )


# ===============================================================
# SECTION D
# SHAP
# ===============================================================

def get_shap_model(model_name):

    if model_name == "LogisticRegression":

        return LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        )

    if model_name == "RandomForest":

        return RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    if model_name == "XGBoost":

        return XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )

    raise ValueError(
        f"Unknown SHAP model: {model_name}"
    )


# ===============================================================
# SHAP TOP-K
# ===============================================================

def compute_shap_topk(
    model,
    model_name,
    X_train,
    y_train,
    X_val,
    feature_cols,
    top_k=TOP_K
):

    model.fit(
        X_train,
        y_train
    )

    # -----------------------------------------------------------
    # Tree models
    # -----------------------------------------------------------

    if model_name in [
        "RandomForest",
        "XGBoost"
    ]:

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = (
            explainer.shap_values(
                X_val
            )
        )

        if isinstance(
            shap_values,
            list
        ):

            shap_values = shap_values[1]

        if shap_values.ndim == 3:

            shap_values = (
                shap_values[:, :, 1]
            )

    # -----------------------------------------------------------
    # Logistic Regression
    # -----------------------------------------------------------

    else:

        background = shap.sample(
            X_train,
            min(
                50,
                len(X_train)
            ),
            random_state=RANDOM_STATE
        )

        explainer = shap.LinearExplainer(
            model,
            background
        )

        shap_values = (
            explainer.shap_values(
                X_val
            )
        )

    mean_abs_shap = (
        np.abs(shap_values)
        .mean(axis=0)
    )

    ranked_features = (
        pd.Series(
            mean_abs_shap,
            index=feature_cols
        )
        .sort_values(
            ascending=False
        )
    )

    return set(
        ranked_features.index[:top_k]
    )


# ===============================================================
# JACCARD STABILITY
# ===============================================================

def jaccard(
    set_a,
    set_b
):

    if not set_a and not set_b:

        return 1.0

    return (
        len(set_a & set_b)
        /
        len(set_a | set_b)
    )


# ===============================================================
# SHAP FOLD STABILITY
# ===============================================================

def fold_stability(
    X,
    y,
    feature_cols,
    model_name,
    seed=RANDOM_STATE
):

    skf = StratifiedKFold(
        n_splits=N_CV_FOLDS,
        shuffle=True,
        random_state=seed
    )

    top_k_sets = []

    for train_idx, val_idx in skf.split(
        X,
        y
    ):

        X_train = X.iloc[
            train_idx
        ]

        X_val = X.iloc[
            val_idx
        ]

        y_train = y[
            train_idx
        ]

        # Need both classes
        if len(np.unique(y_train)) < 2:

            continue

        scaler = RobustScaler()

        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(
                X_train
            ),
            columns=feature_cols
        )

        X_val_scaled = pd.DataFrame(
            scaler.transform(
                X_val
            ),
            columns=feature_cols
        )

        model = get_shap_model(
            model_name
        )

        top_k = compute_shap_topk(
            model=model,
            model_name=model_name,
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            feature_cols=feature_cols
        )

        top_k_sets.append(
            top_k
        )

    if len(top_k_sets) < 2:

        return np.nan

    pairs = list(
        combinations(
            range(len(top_k_sets)),
            2
        )
    )

    scores = [

        jaccard(
            top_k_sets[i],
            top_k_sets[j]
        )

        for i, j in pairs
    ]

    return float(
        np.mean(scores)
    )


# ===============================================================
# SHAP STABILITY + SHUFFLED LABEL CONTROL
# ===============================================================

def run_shap_stability(
    X,
    y,
    feature_cols
):

    results = []

    for model_name in SUPERVISED_MODELS:

        print(
            f"    SHAP stability: {model_name}"
        )

        # -------------------------------------------------------
        # Real labels
        # -------------------------------------------------------

        real_stability = fold_stability(
            X,
            y,
            feature_cols,
            model_name
        )

        # -------------------------------------------------------
        # Shuffled labels
        # -------------------------------------------------------

        shuffled_scores = []

        rng = np.random.RandomState(
            RANDOM_STATE
        )

        for i in range(
            N_SHUFFLES
        ):

            shuffled_y = y.copy()

            rng.shuffle(
                shuffled_y
            )

            stability = fold_stability(
                X,
                shuffled_y,
                feature_cols,
                model_name,
                seed=(
                    RANDOM_STATE + i + 1
                )
            )

            if not np.isnan(
                stability
            ):

                shuffled_scores.append(
                    stability
                )

        if len(
            shuffled_scores
        ) > 0:

            shuffled_mean = float(
                np.mean(
                    shuffled_scores
                )
            )

            shuffled_std = float(
                np.std(
                    shuffled_scores
                )
            )

            percentile = (
                np.mean(
                    np.array(
                        shuffled_scores
                    )
                    <
                    real_stability
                )
                * 100
            )

        else:

            shuffled_mean = np.nan
            shuffled_std = np.nan
            percentile = np.nan

        signal_detected = (
            percentile >= 95
            if not np.isnan(percentile)
            else False
        )

        results.append({

            "model":
                model_name,

            "real_label_stability":
                real_stability,

            "shuffled_label_mean":
                shuffled_mean,

            "shuffled_label_std":
                shuffled_std,

            "real_percentile_vs_shuffled":
                percentile,

            "signal_detected":
                signal_detected
        })

    return pd.DataFrame(
        results
    )


# ===============================================================
# SECTION E
# ERROR ANALYSIS
# ===============================================================

def run_error_analysis(
    base_df,
    feature_cols
):

    positive = base_df[
        base_df["Anomaly Detected"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "yes"
    ].copy()

    negative = base_df[
        base_df["Anomaly Detected"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "no"
    ].copy()

    summary_rows = []

    for col in feature_cols:

        if col == "Pump Status":

            continue

        summary_rows.append({

            "feature":
                col,

            "positive_mean":
                positive[col].mean(),

            "negative_mean":
                negative[col].mean(),

            "positive_std":
                positive[col].std(),

            "negative_std":
                negative[col].std(),

            "mean_difference":
                (
                    positive[col].mean()
                    -
                    negative[col].mean()
                )
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    # -----------------------------------------------------------
    # Pump crosstab
    # -----------------------------------------------------------

    pump_crosstab = pd.crosstab(
        base_df["Pump Status"],
        base_df["Anomaly Detected"]
    )

    # -----------------------------------------------------------
    # Time gaps between anomalies
    # -----------------------------------------------------------

    positive_sorted = (
        positive
        .sort_values("Timestamp")
    )

    time_gaps = (
        positive_sorted["Timestamp"]
        .diff()
        .dt.total_seconds()
        /
        60.0
    )

    time_gap_stats = (
        time_gaps.describe()
    )

    return (
        summary_df,
        pump_crosstab,
        time_gap_stats
    )


# ===============================================================
# FEATURE DISTRIBUTION SUMMARY
# ===============================================================

def create_class_distribution_output(
    y
):

    counts = pd.Series(
        y
    ).value_counts()

    rows = []

    for label in [0, 1]:

        count = int(
            counts.get(
                label,
                0
            )
        )

        rows.append({

            "class":
                (
                    "Normal"
                    if label == 0
                    else "Anomaly"
                ),

            "label":
                label,

            "count":
                count,

            "percentage":
                (
                    count
                    /
                    len(y)
                    *
                    100
                )
        })

    return pd.DataFrame(
        rows
    )


# ===============================================================
# MAIN
# ===============================================================

if __name__ == "__main__":

    # -----------------------------------------------------------
    # Create output directory
    # -----------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nOutput directory:"
        f"\n{OUT_DIR}"
    )

    # ===========================================================
    # LOAD DATA
    # ===========================================================

    df, y = load_and_prepare_dataset()

    # ===========================================================
    # BUILD FEATURES
    # ===========================================================

    print(
        "\nBuilding feature sets..."
    )

    feature_sets, base_df = (
        build_feature_sets(df)
    )

    print(
        "\nFeature sets:"
    )

    for name, (
        X,
        cols
    ) in feature_sets.items():

        print(
            f"  {name}: "
            f"{len(cols)} features"
        )

    # ===========================================================
    # SAVE CLASS DISTRIBUTION
    # ===========================================================

    class_distribution = (
        create_class_distribution_output(
            y
        )
    )

    class_distribution.to_csv(
        OUT_DIR /
        "section_A_class_distribution.csv",
        index=False
    )

    # ===========================================================
    # SECTION B
    # ===========================================================

    print(
        "\n======================================================"
    )

    print(
        "SECTION B: DATA-QUALITY DIAGNOSTICS"
    )

    print(
        "======================================================"
    )

    X_raw, raw_cols = (
        feature_sets["raw"]
    )

    diagnostics = run_diagnostics(
        X_raw,
        y,
        raw_cols
    )

    diagnostics.to_csv(
        OUT_DIR /
        "section_B_diagnostics.csv",
        index=False
    )

    print(
        diagnostics.to_string(
            index=False
        )
    )

    # ===========================================================
    # SECTION C
    # ===========================================================

    print(
        "\n======================================================"
    )

    print(
        "SECTION C: IMBALANCE-AWARE MODEL GRID"
    )

    print(
        "======================================================"
    )

    all_results = []

    for feature_set_name, (
        X,
        feature_cols
    ) in feature_sets.items():

        print(
            f"\nFeature set:"
            f" {feature_set_name}"
        )

        print(
            f"Number of features:"
            f" {len(feature_cols)}"
        )

        # -------------------------------------------------------
        # Raw gets complete grid
        # -------------------------------------------------------

        if feature_set_name == "raw":

            regimes = [
                "none",
                "class_weight",
                "smote",
                "adasyn"
            ]

        # -------------------------------------------------------
        # Engineered features
        # -------------------------------------------------------

        else:

            regimes = [
                "none",
                "class_weight",
                "smote",
                "adasyn"
            ]

        grid_results, _ = (
            run_grid_for_feature_set(
                X=X,
                y=y,
                feature_set_name=feature_set_name,
                feature_cols=feature_cols,
                regimes=regimes,
                model_names=SUPERVISED_MODELS
            )
        )

        all_results.append(
            grid_results
        )

    full_results = pd.concat(
        all_results,
        ignore_index=True
    )

    # -----------------------------------------------------------
    # Sort by F1
    # -----------------------------------------------------------

    full_results = (
        full_results
        .sort_values(
            [
                "F1",
                "PR_AUC",
                "Recall"
            ],
            ascending=False
        )
        .reset_index(drop=True)
    )

    full_results.to_csv(
        OUT_DIR /
        "section_C_full_ablation_grid.csv",
        index=False
    )

    print(
        "\n\nMODEL RESULTS:"
    )

    display_columns = [

        "feature_set",
        "regime",
        "model",
        "ROC_AUC",
        "PR_AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "Balanced_Accuracy"
    ]

    print(
        full_results[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ===========================================================
    # BEST MODEL
    # ===========================================================

    valid_results = (
        full_results
        .dropna(
            subset=[
                "F1",
                "PR_AUC"
            ]
        )
    )

    if len(valid_results) > 0:

        best_model = (
            valid_results
            .sort_values(
                [
                    "F1",
                    "PR_AUC",
                    "Recall"
                ],
                ascending=False
            )
            .iloc[0]
        )

        print(
            "\n======================================================"
        )

        print(
            "BEST CONFIGURATION"
        )

        print(
            "======================================================"
        )

        print(
            f"Feature set : "
            f"{best_model['feature_set']}"
        )

        print(
            f"Regime      : "
            f"{best_model['regime']}"
        )

        print(
            f"Model       : "
            f"{best_model['model']}"
        )

        print(
            f"ROC-AUC     : "
            f"{best_model['ROC_AUC']:.4f}"
        )

        print(
            f"PR-AUC      : "
            f"{best_model['PR_AUC']:.4f}"
        )

        print(
            f"Precision   : "
            f"{best_model['Precision']:.4f}"
        )

        print(
            f"Recall      : "
            f"{best_model['Recall']:.4f}"
        )

        print(
            f"F1          : "
            f"{best_model['F1']:.4f}"
        )

        print(
            f"MCC         : "
            f"{best_model['MCC']:.4f}"
        )

    # ===========================================================
    # SECTION D
    # ===========================================================

    print(
        "\n======================================================"
    )

    print(
        "SECTION D: SHAP STABILITY + SHUFFLED LABEL CONTROL"
    )

    print(
        "======================================================"
    )

    shap_results = run_shap_stability(
        X_raw,
        y,
        raw_cols
    )

    shap_results.to_csv(
        OUT_DIR /
        "section_D_shap_stability_control.csv",
        index=False
    )

    print(
        shap_results.to_string(
            index=False
        )
    )

    # ===========================================================
    # SECTION E
    # ===========================================================

    print(
        "\n======================================================"
    )

    print(
        "SECTION E: ANOMALY CASE ERROR ANALYSIS"
    )

    print(
        "======================================================"
    )

    (
        feature_summary,
        pump_crosstab,
        time_gap_stats
    ) = run_error_analysis(
        base_df,
        raw_cols
    )

    feature_summary.to_csv(
        OUT_DIR /
        "section_E_feature_summary_by_class.csv",
        index=False
    )

    pump_crosstab.to_csv(
        OUT_DIR /
        "section_E_pump_status_crosstab.csv"
    )

    time_gap_stats.to_csv(
        OUT_DIR /
        "section_E_positive_case_time_gaps.csv"
    )

    print(
        "\nFeature summary:"
    )

    print(
        feature_summary.to_string(
            index=False
        )
    )

    print(
        "\nPump Status vs Anomaly:"
    )

    print(
        pump_crosstab
    )

    print(
        "\nTime gap statistics:"
    )

    print(
        time_gap_stats
    )

    # ===========================================================
    # FINAL OUTPUT
    # ===========================================================

    print(
        "\n\n======================================================"
    )

    print(
        "ALL DELIVERABLES SAVED"
    )

    print(
        "======================================================"
    )

    print(
        f"{OUT_DIR}"
    )

    output_files = [

        "section_A_class_distribution.csv",

        "section_B_diagnostics.csv",

        "section_C_full_ablation_grid.csv",

        "section_D_shap_stability_control.csv",

        "section_E_feature_summary_by_class.csv",

        "section_E_pump_status_crosstab.csv",

        "section_E_positive_case_time_gaps.csv"
    ]

    for file_name in output_files:

        print(
            f"  ✓ {file_name}"
        )

    print(
        "\nPipeline completed successfully."
    )
