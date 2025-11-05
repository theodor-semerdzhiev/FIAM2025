import os, json, pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple

BASE_YEARLY_DIR = "qlattice_yearly"  # change if needed

# ---------- public API ----------
def predict_for_year(
    df: pd.DataFrame,
    test_year: int,
    base_yearly_dir: str = "qlattice_yearly",
    scale_to_validation: bool = True,
    return_both: bool = False,
    output_row: str = "stock_ret"
) -> pd.DataFrame:
    """
    Produce predictions for `df` using the best model of `test_year`.

    - Columns with NaNs are NOT dropped; NaNs are replaced with TRAIN means.
    - Missing expected feature columns are added and filled with TRAIN means.
    - Features are standardized with TRAIN mu/sd (from the saved preproc).
    - If `scale_to_validation=True`, outputs are aligned to that year's
      validation prediction distribution via z-score mapping.

    Returns:
        pd.Series of predictions (scaled if enabled) or
        pd.DataFrame with columns ['y_pred_raw', 'y_pred_scaled'] if return_both=True.
    """
    # ---------- loader ----------
    def load_qlattice_model(model_path: str, preproc_path: Optional[str]) -> Tuple[Any, Dict[str, Any]]:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        if not preproc_path:
            raise FileNotFoundError("preproc .npz is required for inference.")
        data = np.load(preproc_path, allow_pickle=True)
        preproc = {
            "feat_cols": data["feat_cols"].tolist(),
            "mu": np.asarray(data["mu"], dtype=float),
            "sd": np.asarray(data["sd"], dtype=float),
        }
        return model, preproc

    # ---------- scaling stats ----------
    def _load_val_scaling_stats(year_dir: str, year: int) -> Tuple[float, float]:
        stats_json = os.path.join(year_dir, f"qlattice_distribution_stats_{year}.json")
        if os.path.exists(stats_json):
            with open(stats_json, "r", encoding="utf-8") as f:
                d = json.load(f)
            return (
                float(d["val_pred_mean"]), 
                float(d["val_pred_std"]), 
                float(d["model_pred_mean"]), 
                float(d["model_pred_std"])
            )
        
        raise FileNotFoundError(stats_json)

    # ---------- preprocessing ----------
    def _ensure_features(df: pd.DataFrame, feat_cols, mu) -> pd.DataFrame:
        """
        Ensure all expected features exist.
        - Add any missing columns, filled with the TRAIN mean for that feature.
        - Replace NaNs with TRAIN mean, then standardize.
        """
        X = df.reindex(columns=feat_cols, copy=False)
        # Add missing columns with mean
        miss = [c for c in feat_cols if c not in X.columns]
        if miss:
            add = pd.DataFrame({c: mu[i] for i, c in enumerate(feat_cols) if c in miss}, index=df.index)
            X = pd.concat([X, add], axis=1)
            X = X[feat_cols]

        # Fill NaNs with TRAIN means (do NOT drop columns)
        # Do it vectorized without per-column loops
        X = X.astype(float, copy=False)
        means_map = {c: float(mu[i]) for i, c in enumerate(feat_cols)}
        X = X.fillna(value=means_map)

        return X

    def _standardize_inplace(X: pd.DataFrame, feat_cols, mu, sd) -> None:
        # sd guards
        sd = np.where(~np.isfinite(sd) | (sd == 0.0), 1.0, sd)
        # in-place z-score
        for i, c in enumerate(feat_cols):
            col = X[c].to_numpy(dtype=float, copy=False)
            np.subtract(col, mu[i], out=col)
            np.divide(col, sd[i], out=col)
            # write back not needed; numpy view writes through

    # ---------- model prediction ----------
    def _predict_safely(model: Any, X: pd.DataFrame) -> np.ndarray:
        # Try X-only first
        try:
            return np.asarray(model.predict(X), dtype=float)
        except Exception:
            # Some QLattice models expect [y] + X; feed a dummy y column
            tmp = pd.concat([pd.Series(0.0, index=X.index, name="__dummy_y__"), X], axis=1)
            return np.asarray(model.predict(tmp), dtype=float)

    year_dir = os.path.join(base_yearly_dir, str(test_year))
    model_path   = os.path.join(year_dir, f"qlattice_model_{test_year}.pkl")
    preproc_path = os.path.join(year_dir, f"qlattice_preproc_{test_year}.npz")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing: {model_path}")
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Missing: {preproc_path}")

    model, preproc = load_qlattice_model(model_path, preproc_path)
    feat_cols, mu, sd = preproc["feat_cols"], preproc["mu"], preproc["sd"]

    # Build X with required features, fill NaNs with TRAIN means, then standardize
    X = _ensure_features(df, feat_cols, mu)
    _standardize_inplace(X, feat_cols, mu, sd)

    # Predict raw
    y_raw = _predict_safely(model, X)

    if not scale_to_validation:
        return pd.Series(y_raw, index=df.index, name="y_pred")

    # Align to validation distribution
    val_mean, val_std, pred_mean, pred_std = _load_val_scaling_stats(year_dir, test_year)

    z = (y_raw - pred_mean) / pred_std
    y_scaled = val_mean + z * (val_std if val_std > 0 else 1.0)

    if return_both:
        return pd.DataFrame(
            {f"{output_row}_raw": y_raw, f"{output_row}_scaled": y_scaled},
            index=df.index,
        )
    else:
        return pd.DataFrame(pd.Series(y_scaled, index=df.index, name=output_row))
