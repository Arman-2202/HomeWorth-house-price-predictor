"""
Train the price model.

    python -m src.train_model          # full run: compares 3 algorithms (~1 min)
    python -m src.train_model --quick  # only the final model (~10 sec)

Outputs:
    models/price_model.joblib   - fitted scikit-learn pipeline
    models/metrics.json         - accuracy, model comparison, feature importance
    models/holdout_sample.csv   - actual vs predicted on unseen data (for charts)
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .data_generator import generate


def _preprocessor(scale=False):
    num = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), C.CAT_FEATURES),
         ("num", num, C.NUM_FEATURES)],
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")
    return pre


def _wrap(regressor):
    return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)


def build_final_model():
    """Gradient boosting with sensible monotonic rules (more area => never cheaper)."""
    mono = {"area_sqft": 1, "locality_rate": 1, "age_years": -1, "metro_km": -1,
            "rera": 1, "parking": 1, "near_social": 1, "modular_kitchen": 1, "am_pool": 1, "am_gym": 1, "am_clubhouse": 1,
            "am_security": 1, "am_power_backup": 1, "am_garden": 1, "am_gated": 1,
            "am_play_area": 1}
    hgb = HistGradientBoostingRegressor(
        max_iter=700, learning_rate=0.06, max_depth=8, min_samples_leaf=25,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=40, monotonic_cst=mono, random_state=42,
    )
    return Pipeline([("prep", _preprocessor()), ("model", _wrap(hgb))])


def build_baselines():
    return {
        "Linear Regression (Ridge)": Pipeline([("prep", _preprocessor(scale=True)),
                                               ("model", _wrap(Ridge(alpha=1.0)))]),
        "Random Forest": Pipeline([("prep", _preprocessor()),
                                   ("model", _wrap(RandomForestRegressor(
                                       n_estimators=120, min_samples_leaf=3, max_features=0.6,
                                       n_jobs=-1, random_state=42)))]),
    }


def _metrics(y_true, y_pred):
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae_lakh": float(mean_absolute_error(y_true, y_pred) / 1e5),
        "mape_pct": float(ape.mean() * 100),
        "median_ape_pct": float(np.median(ape) * 100),
        "within_10_pct": float((ape <= 0.10).mean() * 100),
        "within_15_pct": float((ape <= 0.15).mean() * 100),
    }


def train(quick=False, verbose=True):
    t0 = time.time()
    if not C.DATA_FILE.exists():
        if verbose:
            print("No dataset found - generating one ...")
        C.DATA_DIR.mkdir(parents=True, exist_ok=True)
        generate().to_csv(C.DATA_FILE, index=False)

    df = pd.read_csv(C.DATA_FILE)
    X, y = df[C.FEATURES], df[C.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=7,
                                              stratify=df["city"])
    if verbose:
        print(f"Training on {len(X_tr):,} rows, testing on {len(X_te):,} unseen rows")

    # keep the previous comparison table when doing a quick retrain
    comparison = []
    if quick and C.METRICS_FILE.exists():
        try:
            comparison = json.loads(C.METRICS_FILE.read_text()).get("comparison", [])
        except Exception:
            comparison = []
    if not quick:
        for name, pipe in build_baselines().items():
            pipe.fit(X_tr, y_tr)
            m = _metrics(y_te.values, pipe.predict(X_te))
            comparison.append({"model": name, **m})
            if verbose:
                print(f"  {name:28s} R2={m['r2']:.3f}  MAPE={m['mape_pct']:.1f}%")

    final = build_final_model()
    final.fit(X_tr, y_tr)
    pred = final.predict(X_te)
    final_m = _metrics(y_te.values, pred)
    comparison = [c for c in comparison if not c["model"].startswith("Gradient Boosting")]
    comparison.append({"model": "Gradient Boosting (final)", **final_m})
    if verbose:
        print(f"  {'Gradient Boosting (final)':28s} R2={final_m['r2']:.3f}  MAPE={final_m['mape_pct']:.1f}%")

    # residual spread -> prediction interval (80% band)
    log_res = np.log(y_te.values / pred)
    q_lo, q_hi = np.quantile(log_res, [0.10, 0.90])

    # permutation importance on original columns (amenities grouped)
    sample = X_te.sample(min(3000, len(X_te)), random_state=1)
    imp = permutation_importance(final, sample, y_te.loc[sample.index], n_repeats=3,
                                 random_state=1, scoring="neg_mean_absolute_percentage_error",
                                 n_jobs=1)
    imp_s = pd.Series(imp.importances_mean, index=C.FEATURES).clip(lower=0)
    grouped = imp_s.drop(C.AMENITY_COLS)
    grouped["amenities"] = imp_s[C.AMENITY_COLS].sum()
    grouped = (grouped / grouped.sum() * 100).sort_values(ascending=False)
    importance = {C.FEATURE_LABELS.get(k, k): round(float(v), 2) for k, v in grouped.items()}

    # accuracy by city (for the insights page)
    by_city = (pd.DataFrame({"city": X_te["city"].values, "y": y_te.values, "p": pred})
               .assign(ape=lambda d: (d.p - d.y).abs() / d.y)
               .groupby("city")["ape"].mean().mul(100).round(2).to_dict())

    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, C.MODEL_FILE, compress=3)

    hold = X_te.copy()
    hold["actual"] = y_te.values
    hold["predicted"] = pred
    hold.sample(min(2000, len(hold)), random_state=3).to_csv(C.HOLDOUT_FILE, index=False)

    meta = {
        "sklearn_version": sklearn.__version__,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "n_cities": len(C.CITIES),
        "n_localities": int(sum(len(c["localities"]) for c in C.CITIES.values())),
        "final": final_m,
        "interval_log": [float(q_lo), float(q_hi)],
        "comparison": comparison,
        "importance": importance,
        "mape_by_city": by_city,
        "trained_seconds": round(time.time() - t0, 1),
    }
    C.METRICS_FILE.write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"Done in {meta['trained_seconds']}s  |  80% band: {np.exp(q_lo):.2f}x - {np.exp(q_hi):.2f}x")
    return final, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="train only the final model")
    args = ap.parse_args()
    train(quick=args.quick)
