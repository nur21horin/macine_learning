"""
FULL PIPELINE: Imbalance-Aware ML Anomaly Detection in NPP CPS Data
=====================================================================
Consolidates every deliverable from the methodology into one runnable script.

SECTIONS
  A. Feature engineering       -> raw / temporal / physics-informed feature sets
  B. Data-quality diagnostics  -> correlation + mutual information audit
  C. Model x Regime x FeatureSet grid (the ablation study)
       6 models x 4 imbalance regimes x 3 feature sets, with bootstrap CIs
  D. SHAP stability + shuffled-label control (core novel diagnostic)
  E. Error analysis of the positive ("anomaly") cases

All outputs are saved as CSVs to ./outputs/ — these map directly
onto the Results section tables/figures described in the paper guide.

Usage: python full_pipeline.py
Requires: pandas, numpy, scikit-learn, imbalanced-learn, xgboost, shap, scipy
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, brier_score_loss, mutual_info_score
)
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from scipy.stats import norm
import shap

RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH ="./Nuclear_Power_Plant_CPS_Dataset.csv"
OUT_DIR = ROOT_DIR / "outputs"
N_BOOTSTRAP = 200         
N_CV_FOLDS = 5
N_SHUFFLES = 8            
TOP_K = 5

SUPERVISED = {"LogisticRegression", "RandomForest", "XGBoost"}
UNSUPERVISED = {"IsolationForest", "LOF", "OneClassSVM"}


# =====================================================================
# SECTION A: Feature Engineering
# =====================================================================
def build_feature_sets(df):
    """Returns dict of {feature_set_name: (X_df, feature_cols)}."""
    base = df.copy()
    base["Pump Status"] = (base["Pump Status"] == "ON").astype(int)
    base["Timestamp"] = pd.to_datetime(base["Timestamp"])
    base = base.sort_values("Timestamp").reset_index(drop=True)

    raw_cols = [c for c in df.columns if c not in ("Timestamp", "Anomaly Detected")]
    sets = {}

    # --- Raw ---
    sets["raw"] = (base[raw_cols].copy(), list(raw_cols))

    # --- Temporal (rolling stats over 5/10/15-minute windows) ---
    temporal = base[raw_cols].copy()
    sensor_cols = [c for c in raw_cols if c != "Pump Status"]
    for w in (5, 10, 15):
        for c in sensor_cols:
            temporal[f"{c}_roll{w}_mean"] = base[c].rolling(w, min_periods=1).mean()
            temporal[f"{c}_roll{w}_std"] = base[c].rolling(w, min_periods=1).std().fillna(0)
        temporal[f"delta_{w}"] = 0  # placeholder namespace guard (avoids collision below)
    for c in sensor_cols:
        temporal[f"{c}_delta1"] = base[c].diff().fillna(0)
    temporal = temporal.drop(columns=[c for c in temporal.columns if c.startswith("delta_")])
    sets["temporal"] = (temporal, list(temporal.columns))

    # --- Physics-informed composite features ---
    physics = base[raw_cols].copy()
    # Energy-balance-style residual: Power Output vs. a naive linear expectation from Steam Flow & Turbine Speed
    # (illustrative physical constraint, not a calibrated plant model)
    expected_power = 0.5 * base["Steam Flow Rate (kg/s)"] + 0.01 * base["Turbine Speed (RPM)"]
    physics["power_residual"] = base["Power Output (MW)"] - expected_power
    # Pressure-temperature deviation from a naive expected relationship
    expected_pressure = 0.02 * base["Reactor Temp (°C)"]
    physics["pressure_temp_residual"] = base["Pressure (MPa)"] - expected_pressure
    # Coolant adequacy ratio
    physics["coolant_per_power"] = base["Coolant Flow Rate (L/s)"] / (base["Power Output (MW)"].replace(0, np.nan))
    physics["coolant_per_power"] = physics["coolant_per_power"].fillna(0)
    sets["physics"] = (physics, list(physics.columns))

    return sets, base


# =====================================================================
# SECTION B: Data-Quality Diagnostics (correlation + mutual information)
# =====================================================================
def run_diagnostics(X, y, feature_cols):
    rows = []
    for c in feature_cols:
        corr = np.corrcoef(X[c].values, y)[0, 1]
        rows.append({"feature": c, "pearson_corr": corr})
    corr_df = pd.DataFrame(rows).sort_values("pearson_corr", key=abs, ascending=False)

    mi = mutual_info_classif(X[feature_cols], y, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"feature": feature_cols, "mutual_info": mi}).sort_values(
        "mutual_info", ascending=False
    )

    diag = corr_df.merge(mi_df, on="feature")
    return diag


# =====================================================================
# SECTION C: Model x Regime x FeatureSet Grid
# =====================================================================
def apply_regime(X_train, y_train, regime, random_state=RANDOM_STATE):
    if regime == "none":
        return X_train, y_train
    elif regime == "class_weight":
        return X_train, y_train  # weighting applied at model level
    elif regime == "smote":
        k = min(5, max(1, (y_train == 1).sum() - 1))
        sm = SMOTE(random_state=random_state, k_neighbors=k)
        return sm.fit_resample(X_train, y_train)
    elif regime == "adasyn":
        k = min(5, max(1, (y_train == 1).sum() - 1))
        ad = ADASYN(random_state=random_state, n_neighbors=k)
        return ad.fit_resample(X_train, y_train)
    raise ValueError(regime)


def get_models(class_weight=None, scale_pos_weight=None):
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight=class_weight, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=250, class_weight=class_weight, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
            scale_pos_weight=scale_pos_weight if scale_pos_weight else 1,
        ),
        "IsolationForest": IsolationForest(
            n_estimators=250, random_state=RANDOM_STATE, contamination="auto"
        ),
        "LOF": LocalOutlierFactor(n_neighbors=20, novelty=True, contamination="auto"),
        "OneClassSVM": OneClassSVM(kernel="rbf", nu=0.05, gamma="scale"),
    }


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    stats = []
    for _ in range(n_boot):
        bi = np.concatenate([
            rng.choice(idx_pos, size=len(idx_pos), replace=True),
            rng.choice(idx_neg, size=len(idx_neg), replace=True),
        ])
        try:
            stats.append(metric_fn(y_true[bi], y_score[bi]))
        except Exception:
            continue
    stats = np.array(stats)
    if len(stats) == 0:
        return np.nan, np.nan, np.nan
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(np.mean(stats)), float(lo), float(hi)


def get_scores(model, name, X_test):
    if name in SUPERVISED:
        return model.predict_proba(X_test)[:, 1]
    return -model.decision_function(X_test)


def get_preds(model, name, X_test, scores):
    if name in SUPERVISED:
        return (scores >= 0.5).astype(int)
    raw = model.predict(X_test)
    return (raw == -1).astype(int)


def run_grid_for_feature_set(X, y, feature_set_name, feature_cols,
                              regimes=None, model_names=None):
    regimes = regimes or ["none", "class_weight", "smote", "adasyn"]
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1765,
        stratify=y_train_full, random_state=RANDOM_STATE
    )

    scaler = RobustScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=feature_cols)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    results = []
    for regime in regimes:
        if regime == "class_weight":
            Xr, yr = X_train_s, y_train
            cw = "balanced"
            spw = (yr == 0).sum() / max((yr == 1).sum(), 1)
        else:
            Xr, yr = apply_regime(X_train_s, y_train, regime)
            cw, spw = None, None

        models = get_models(class_weight=cw, scale_pos_weight=spw)
        if model_names is not None:
            models = {k: v for k, v in models.items() if k in model_names}
        for name, model in models.items():
            try:
                if name in SUPERVISED:
                    model.fit(Xr, yr)
                else:
                    normal_only = Xr[np.array(yr) == 0]
                    model.fit(normal_only)

                scores = get_scores(model, name, X_test_s.values)
                preds = get_preds(model, name, X_test_s.values, scores)

                roc = roc_auc_score(y_test, scores)
                pr = average_precision_score(y_test, scores)
                f1 = f1_score(y_test, preds, zero_division=0)
                mcc = matthews_corrcoef(y_test, preds)
                brier = brier_score_loss(y_test, np.clip(scores, 0, 1)) if name in SUPERVISED else np.nan

                pr_m, pr_lo, pr_hi = bootstrap_ci(y_test, scores, average_precision_score)
                roc_m, roc_lo, roc_hi = bootstrap_ci(y_test, scores, roc_auc_score)

                results.append({
                    "feature_set": feature_set_name, "regime": regime, "model": name,
                    "ROC_AUC": roc, "ROC_AUC_CI_lo": roc_lo, "ROC_AUC_CI_hi": roc_hi,
                    "PR_AUC": pr, "PR_AUC_CI_lo": pr_lo, "PR_AUC_CI_hi": pr_hi,
                    "F1": f1, "MCC": mcc, "Brier": brier,
                })
            except Exception as e:
                results.append({
                    "feature_set": feature_set_name, "regime": regime, "model": name,
                    "ROC_AUC": np.nan, "PR_AUC": np.nan, "F1": np.nan, "MCC": np.nan,
                    "Brier": np.nan, "error": str(e),
                })
    return pd.DataFrame(results), (X_test_s, y_test)


# =====================================================================
# SECTION D: SHAP Stability + Shuffled-Label Control (on raw feature set)
# =====================================================================
def get_shap_model(name):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    elif name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    elif name == "XGBoost":
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              eval_metric="logloss", random_state=RANDOM_STATE,
                              n_jobs=-1, verbosity=0)


def compute_shap_topk(model, name, X_train, y_train, X_val, feature_cols, top_k=TOP_K):
    model.fit(X_train, y_train)
    if name in ("RandomForest", "XGBoost"):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_val)
        if isinstance(sv, list):
            sv = sv[1]
        if sv.ndim == 3:
            sv = sv[:, :, 1]
    else:
        bg = shap.sample(X_train, min(50, len(X_train)), random_state=RANDOM_STATE)
        explainer = shap.LinearExplainer(model, bg)
        sv = explainer.shap_values(X_val)
    mean_abs = np.abs(sv).mean(axis=0)
    ranked = pd.Series(mean_abs, index=feature_cols).sort_values(ascending=False)
    return set(ranked.index[:top_k])


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def fold_stability(X, y, feature_cols, model_name, seed=RANDOM_STATE):
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
    topk_sets = []
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        if y_tr.sum() < 2:
            continue
        scaler = RobustScaler().fit(X_tr)
        X_tr_s = pd.DataFrame(scaler.transform(X_tr), columns=feature_cols)
        X_val_s = pd.DataFrame(scaler.transform(X_val), columns=feature_cols)
        model = get_shap_model(model_name)
        topk = compute_shap_topk(model, model_name, X_tr_s, y_tr, X_val_s, feature_cols)
        topk_sets.append(topk)
    if len(topk_sets) < 2:
        return np.nan
    pairs = list(combinations(range(len(topk_sets)), 2))
    return float(np.mean([jaccard(topk_sets[i], topk_sets[j]) for i, j in pairs]))


def run_shap_stability(X, y, feature_cols):
    results = []
    for model_name in ["LogisticRegression", "RandomForest", "XGBoost"]:
        real_stab = fold_stability(X, y, feature_cols, model_name)

        shuffled = []
        rng = np.random.RandomState(RANDOM_STATE)
        for i in range(N_SHUFFLES):
            y_shuf = y.copy()
            rng.shuffle(y_shuf)
            s = fold_stability(X, y_shuf, feature_cols, model_name, seed=RANDOM_STATE + i)
            if not np.isnan(s):
                shuffled.append(s)

        shuf_mean, shuf_std = float(np.mean(shuffled)), float(np.std(shuffled))
        percentile = (np.array(shuffled) < real_stab).mean() * 100

        results.append({
            "model": model_name,
            "real_label_stability": real_stab,
            "shuffled_label_mean": shuf_mean,
            "shuffled_label_std": shuf_std,
            "real_percentile_vs_shuffled": percentile,
            "signal_detected": percentile >= 95,
        })
    return pd.DataFrame(results)


# =====================================================================
# SECTION E: Error Analysis of Positive Cases
# =====================================================================
def run_error_analysis(base_df, feature_cols):
    pos = base_df[base_df["Anomaly Detected"] == "Yes"].copy()
    neg = base_df[base_df["Anomaly Detected"] == "No"].copy()

    summary_rows = []
    for c in feature_cols:
        if c == "Pump Status":
            continue
        summary_rows.append({
            "feature": c,
            "positive_mean": pos[c].mean(),
            "negative_mean": neg[c].mean(),
            "positive_std": pos[c].std(),
            "negative_std": neg[c].std(),
        })
    summary_df = pd.DataFrame(summary_rows)

    pump_crosstab = pd.crosstab(base_df["Pump Status"], base_df["Anomaly Detected"])

    pos_sorted = pos.sort_values("Timestamp")
    time_gaps = pos_sorted["Timestamp"].diff().dt.total_seconds() / 60.0

    return summary_df, pump_crosstab, time_gaps.describe()


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")
    print("Loading data and building feature sets...")
    df = pd.read_csv(DATA_PATH)
    y = (df["Anomaly Detected"] == "Yes").astype(int).values
    feature_sets, base_df = build_feature_sets(df)

    # --- Section B: diagnostics (raw feature set) ---
    print("\nSection B: Running data-quality diagnostics (correlation + mutual information)...")
    X_raw, raw_cols = feature_sets["raw"]
    diag_df = run_diagnostics(X_raw, y, raw_cols)
    diag_df.to_csv(OUT_DIR / "section_B_diagnostics.csv", index=False)
    print(diag_df.to_string(index=False))

    # --- Section C: model x regime x feature-set grid (ablation) ---
    print("\nSection C: Running model x regime x feature-set grid (this may take a few minutes)...")
    all_grid_results = []
    for fs_name, (X_fs, cols_fs) in feature_sets.items():
        print(f"  Feature set: {fs_name} ({len(cols_fs)} features)")
        # Scope reduction for runtime: raw gets the full 6-model x 4-regime grid.
        # temporal/physics (ablation sets) get the 3 supervised models x 2 regimes
        # (none, smote) -- sufficient to test whether engineered features recover
        # signal; widen back to the full grid for the final paper run if compute allows.
        if fs_name == "raw":
            grid_df, _ = run_grid_for_feature_set(X_fs, y, fs_name, cols_fs)
        else:
            grid_df, _ = run_grid_for_feature_set(
                X_fs, y, fs_name, cols_fs,
                regimes=["none", "smote"], model_names=list(SUPERVISED)
            )
        all_grid_results.append(grid_df)
    full_grid_df = pd.concat(all_grid_results, ignore_index=True)
    full_grid_df.to_csv(OUT_DIR / "section_C_full_ablation_grid.csv", index=False)
    print(f"  Saved {len(full_grid_df)} rows to section_C_full_ablation_grid.csv")

    # --- Section D: SHAP stability + shuffled-label control (raw feature set) ---
    print("\nSection D: Running SHAP stability + shuffled-label control...")
    shap_df = run_shap_stability(X_raw, y, raw_cols)
    shap_df.to_csv(OUT_DIR / "section_D_shap_stability_control.csv", index=False)
    print(shap_df.to_string(index=False))

    # --- Section E: error analysis ---
    print("\nSection E: Running error analysis of positive cases...")
    summary_df, pump_crosstab, time_gap_stats = run_error_analysis(base_df, raw_cols)
    summary_df.to_csv(OUT_DIR / "section_E_feature_summary_by_class.csv", index=False)
    pump_crosstab.to_csv(OUT_DIR / "section_E_pump_status_crosstab.csv")
    time_gap_stats.to_csv(OUT_DIR / "section_E_positive_case_time_gaps.csv")
    print(summary_df.to_string(index=False))
    print("\nPump Status vs Anomaly crosstab:")
    print(pump_crosstab)

    print(f"\n\n=== ALL DELIVERABLES SAVED TO {OUT_DIR} ===")
    print("section_B_diagnostics.csv")
    print("section_C_full_ablation_grid.csv")
    print("section_D_shap_stability_control.csv")
    print("section_E_feature_summary_by_class.csv")
    print("section_E_pump_status_crosstab.csv")
    print("section_E_positive_case_time_gaps.csv")