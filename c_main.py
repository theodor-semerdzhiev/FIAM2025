import datetime
import time
import pandas as pd
import numpy as np
import os
import gc
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost.callback import EarlyStopping

# --- third-party boosters ---
try:
    from xgboost import XGBRegressor
except Exception as e:
    raise ImportError("XGBoost is required. Install with: pip install xgboost") from e

try:
    from catboost import CatBoostRegressor, Pool
except Exception as e:
    raise ImportError("CatBoost is required. Install with: pip install catboost") from e


# =============================
# helpers
# =============================

def ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def hrule(char="=", n=80):
    return char * n


def reduce_mem_usage(df, verbose=True):
    """Iterate through all the columns of a dataframe and modify the data type to reduce memory usage."""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and not is_datetime64_any_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage reduced to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

from pandas.api.types import is_datetime64_any_dtype

def parse_date_col(s: pd.Series) -> pd.Series:
    """Parse a 'date' column once, robustly, and normalize to naive midnight."""
    if np.issubdtype(s.dtype, np.datetime64):
        out = pd.to_datetime(s, utc=False)
        if hasattr(out.dt, "tz_localize"):
            try:
                out = out.dt.tz_localize(None)
            except TypeError:
                pass
        return out.dt.normalize()

    if np.issubdtype(s.dtype, np.integer):
        s_str = s.astype(str)
        try:
            return pd.to_datetime(s_str, format="%Y%m%d").dt.normalize()
        except Exception:
            out = pd.to_datetime(s_str, format="%Y%m").dt.to_period("M").dt.to_timestamp("M")
            return out.dt.normalize()

    s_str = s.astype(str)
    try:
        out = pd.to_datetime(s_str, format="%Y-%m-%d", errors="raise")
        return out.dt.normalize()
    except Exception:
        pass

    try:
        out = pd.to_datetime(s_str, format="%Y-%m", errors="raise").dt.to_period("M").dt.to_timestamp("M")
        return out.dt.normalize()
    except Exception:
        pass

    try:
        out = pd.to_datetime(s_str, errors="raise")
        if len(out) and out.dt.is_month_start.all():
            out = out.dt.to_period("M").dt.to_timestamp("M")
        return out.dt.normalize()
    except Exception as e:
        probe = pd.to_datetime(s_str, errors="coerce")
        bad = s_str[probe.isna()].head(5).tolist()
        raise ValueError(f"Could not parse 'date'. Examples of bad values: {bad}") from e


# =============================
# Feature Engineering (leak-safe)
# =============================

@dataclass
class FeatureConfig:
    regime_vars: List[str]
    regime_k: int = 4
    regime_pca_components: int = 6
    momentum_vars: Optional[List[str]] = None
    momentum_window: int = 6  # months
    path_window: int = 6      # months (rolling over each id)


def add_meta_features_v2(df, date_col, id_col, regime_col=None):
    df = df.sort_values([id_col, date_col]).copy()

    # Factor momentum
    for col in ["qmj", "o_score"]:
        df[f"{col}_chg6m"] = df.groupby(id_col)[col].diff(6)

    # Factor volatility (rolling instability)
    for col in ["qmj", "ivol_capm_21d"]:
        df[f"{col}_vol6m"] = df.groupby(id_col)[col].rolling(6, min_periods=3).std().reset_index(level=0, drop=True)

    # Peer-relative quality
    df["gp_at_rel"] = df["gp_at"] - df.groupby([date_col, "excntry"])["gp_at"].transform("median")

    # Value × Quality interaction
    df["val_qual_combo"] = df["be_me"] * df["qmj_prof"]

    # Drawdown on 12m return
    roll_max = df.groupby(id_col)["ret_12_1"].rolling(12, min_periods=3).max().reset_index(level=0, drop=True)
    df["dd_12m"] = df["ret_12_1"] - roll_max

    # Persistence: share of last 12 months with positive returns
    df["mom_persistence"] = df.groupby(id_col)["ret_1_0"].rolling(12, min_periods=6).apply(lambda s: (s > 0).mean(), raw=True).reset_index(level=0, drop=True)

    # Signal breadth (multi-factor agreement)
    bullish_flags = [
        df["ret_12_1"] > 0,
        df["qmj"] > 0,
        df["be_me"] > 0,
        df["mispricing_perf"] > 0
    ]
    df["signal_breadth"] = np.sum(bullish_flags, axis=0)

    # Optional: factor × regime interaction
    if regime_col and regime_col in df.columns:
        df["qmj_regime_int"] = df["qmj"] * df[regime_col]

    return df



def add_price_path_features(df, id_col, date_col, ret_col, path_window):
    df = df.sort_values([id_col, date_col]).copy()

    # Use only info available by end of month t:
    # r_prev(t) = realized return for month t (which sits on the row dated t-1 in your schema),
    # so shift stock_ret back to align past realized returns with current row.
    r_prev = df.groupby(id_col)[ret_col].shift(1).astype('float32')
    r_prev = r_prev.fillna(0.0)  # or leave NaN and handle with min_periods>1 below

    # cumulative return over window
    log1p_r = np.log1p(r_prev.clip(lower=-0.999999))
    roll_logsum = log1p_r.groupby(df[id_col]).rolling(path_window, min_periods=2).sum().reset_index(level=0, drop=True)
    cumret = np.expm1(roll_logsum).astype('float32')

    # realized volatility / downside over window
    vol = r_prev.groupby(df[id_col]).rolling(path_window, min_periods=2).std().reset_index(level=0, drop=True).astype('float32')
    r_down = r_prev.clip(upper=0.0)
    downside = r_down.groupby(df[id_col]).rolling(path_window, min_periods=2).std().reset_index(level=0, drop=True).astype('float32')

    # sign-changes (choppiness)
    sgn = np.sign(r_prev.replace(0.0, 0.0))
    sgn_shift = sgn.groupby(df[id_col]).shift(1)
    signchg_flag = (sgn != sgn_shift).astype('float32')
    sign_changes = signchg_flag.groupby(df[id_col]).rolling(path_window, min_periods=2).sum().reset_index(level=0, drop=True).astype('float32')

    # simple path score
    path_score = cumret / (1.0 + vol.fillna(0.0) + 0.25 * sign_changes.fillna(0.0))
    path_score = path_score.astype('float32')

    df["pp_cumret_w"]   = cumret.values
    df["pp_vol_w"]       = vol.values
    df["pp_downside_w"] = downside.values
    df["pp_signchg_w"]  = sign_changes.values
    df["price_path"]    = path_score.values
    return df




def compute_factor_ic_by_date(df: pd.DataFrame, date_col: str, ret_col: str, vars_: List[str]) -> pd.DataFrame:
    """Cross-sectional Spearman IC for each factor at each date (using same-date returns).
    We'll use these only from *past* dates to form a momentum score for the *next* date.
    """
    out = []
    for dt, chunk in df.groupby(date_col, sort=True):
        row = {date_col: dt}
        y = chunk[ret_col]
        for v in vars_:
            x = chunk[v]
            if x.notna().sum() >= 10 and y.notna().sum() >= 10:
                row[f"IC_{v}"] = x.rank(pct=True).corr(y.rank(pct=True))
            else:
                row[f"IC_{v}"] = np.nan
        out.append(row)
    return pd.DataFrame(out).sort_values(date_col)


def add_factor_momentum(df: pd.DataFrame, date_col: str, ret_col: str, vars_: List[str], window: int) -> pd.DataFrame:
    """Create a single momentum score summarizing recent factor efficacy.
    For each date t, we compute the rolling mean of factor ICs over the last `window` months
    (using dates strictly < t), then average across selected vars. The resulting scalar is
    aligned to date t and broadcast to each row on that date.
    """
    ic_df = compute_factor_ic_by_date(df, date_col, ret_col, vars_)
    # rolling mean over past window, closed='left' via shift(1)
    ic_roll = ic_df.set_index(date_col).shift(1).rolling(window=window, min_periods=max(2, window//2)).mean()
    ic_roll["factor_momentum"] = ic_roll[[c for c in ic_roll.columns if c.startswith("IC_")]].mean(axis=1)
    fm = ic_roll[["factor_momentum"]].reset_index()
    return df.merge(fm, on=date_col, how="left")


def _kmeans_assign_manual(Z: np.ndarray, centers: np.ndarray) -> np.ndarray:
    Z = Z.astype(np.float32, copy=False)
    C = centers.astype(np.float32, copy=False)
    z_sq = np.einsum('ij,ij->i', Z, Z)
    c_sq = np.einsum('ij,ij->i', C, C)
    d = z_sq[:, None] - 2.0 * (Z @ C.T) + c_sq[None, :]
    return np.argmin(d, axis=1)

def fit_predict_regime(train_df: pd.DataFrame,
                       apply_df: pd.DataFrame,
                       vars_: List[str],
                       k: int,
                       pca_components: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    # Standardize on train only
    Xtr_raw = train_df[vars_].astype(np.float32)
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw).astype(np.float32)

    # PCA then KMeans
    p = min(pca_components, Xtr.shape[1])
    pca = PCA(n_components=p)  # random_state unused unless randomized solver
    Ztr = pca.fit_transform(Xtr).astype(np.float32)

    # NOTE: no 'algorithm=' here (let sklearn pick a valid one)
    km = KMeans(n_clusters=k, n_init=10, random_state=2025)
    km.fit(Ztr)

    # Transform apply set
    Xap = scaler.transform(apply_df[vars_].astype(np.float32)).astype(np.float32)
    Zap = pca.transform(Xap).astype(np.float32)

    # Assign regimes without KMeans.predict (avoids threadpoolctl path)
    if apply_df.shape == train_df.shape and apply_df.index.equals(train_df.index):
        regimes = km.labels_
    else:
        regimes = _kmeans_assign_manual(Zap, km.cluster_centers_)

    out = apply_df.copy()
    out["regime"] = regimes
    dummies = pd.get_dummies(out["regime"], prefix="regime")
    out = pd.concat([out, dummies], axis=1)

    artifacts = {"scaler": scaler, "pca": pca, "kmeans": km}
    return out, artifacts


# =============================
# Main
# =============================

if __name__ == "__main__":
    t0 = time.perf_counter()
    print(hrule())
    print(f"[{ts()}] START")

    pd.set_option("mode.chained_assignment", None)

    # --- paths ---
    work_dir = r"C:\_Files\Personal\Projects\FIAM\FIAM2025"
    print(f"[{ts()}] Working directory: {work_dir}")

    # --- Define essential variables ---
    file_path = os.path.join(work_dir, "data/factor_char_list.csv")
    print(f"[{ts()}] Reading factor list: {file_path}")
    stock_vars = list(pd.read_csv(file_path)["variable"].values)
    print(f"[{ts()}] #factors: {len(stock_vars)}")

    ret_var = "stock_ret"
    id_col = "id"
    date_col = "date"

    # --- Load the pre-processed data directly ---
    rankscaled_path = os.path.join(work_dir, "data/rank_scaled.parquet")
    print(f"[{ts()}] Loading pre-processed data directly from: {rankscaled_path}")
    data = pd.read_parquet(rankscaled_path, engine="pyarrow")
    
    # --- MEMORY OPTIMIZATION: Downcast data types ---
    data = reduce_mem_usage(data)
    
    print(f"[{ts()}] Loaded cached data, shape={data.shape}")

    if data.empty:
        raise RuntimeError("Loaded dataset is empty.")
    print(f"[{ts()}] Date coverage: {data[date_col].min().date()} → {data[date_col].max().date()}")
    
    # --- Config (still needed for model training) ---
    cfg = FeatureConfig(
        regime_vars=stock_vars,
        regime_k=4,
        regime_pca_components=min(6, len(stock_vars)),
        momentum_vars=stock_vars[:12],
        momentum_window=6,
        path_window=6,
    )

    # --- Define feature lists ---
    engineered_cols_static = [
        "price_path", "pp_cumret_w", "pp_vol_w", "pp_downside_w", "pp_signchg_w", "factor_momentum"
    ]
    external_extra_features = [
        "risk_score",
        "similarity_score","similarity_confidence","similarity_matches",
        "similarity_volatility","similarity_strength","similarity_rank_pct",
        "similarity_vs_median","high_confidence_similarity", "mgmt_pred"
    ]
    base_features = stock_vars + engineered_cols_static + external_extra_features

    # =========================
    # walk-forward loop
    # =========================
    print(hrule())
    print(f"[{ts()}] BEGIN walk-forward loop (expanding train, yearly retrain)")

    starting = pd.to_datetime("2005-01-01")
    counter = 0
    loop_iter = 0
    
    out_path = os.path.join(work_dir, "output.csv")
    # MEMORY OPTIMIZATION: Clear the output file before starting
    if os.path.exists(out_path):
        os.remove(out_path)

    def _append_preds(df: pd.DataFrame, path: str):
        """Append predictions to CSV with header only if file is new/empty."""
        desired = ["year","month","ret_eom", id_col, ret_var, date_col, "ridge","xgb","cat","blend",
                   "iter","train_start","train_end","val_end","test_end"]
        for c in desired:
            if c not in df.columns:
                df[c] = np.nan
        df = df[desired]
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        df.to_csv(path, mode="a", header=write_header, index=False)

    while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime("2026-01-01"):

        loop_iter += 1
        if loop_iter < 1:
            counter += 1
            continue

        c0 = starting
        c1 = starting + pd.DateOffset(years=8 + counter)
        c2 = starting + pd.DateOffset(years=10 + counter)
        c3 = starting + pd.DateOffset(years=11 + counter)

        print(hrule())
        print(f"[{ts()}] ITER {loop_iter} | counter={counter}")
        print(f"[{ts()}] Train:      [{c0.date()} → {(c1 - pd.Timedelta(days=1)).date()} ]")
        print(f"[{ts()}] Validate: [ {c1.date()} → {(c2 - pd.Timedelta(days=1)).date()} ]")
        print(f"[{ts()}] Test:       [ {c2.date()} → {(c3 - pd.Timedelta(days=1)).date()} ]")

        date_s = data[date_col]
        train_df    = data[(date_s >= c0) & (date_s < c1)].copy()
        validate_df = data[(date_s >= c1) & (date_s < c2)].copy()
        test_df     = data[(date_s >= c2) & (date_s < c3)].copy()

        print(f"[{ts()}] Split sizes -> train: {len(train_df):,} | validate: {len(validate_df):,} | test: {len(test_df):,}")
        if len(train_df) == 0 or len(validate_df) == 0 or len(test_df) == 0:
            print(f"[{ts()}] WARNING: One or more splits are empty. Skipping this iteration.")
            counter += 1
            continue

        # --- Regime (fit on train only) ---
        print(f"[{ts()}] Fitting REGIME model (PCA+KMeans, k={cfg.regime_k}) on training…")
        # MEMORY OPTIMIZATION: Overwrite dataframes instead of creating new ones
        train_df, artifacts = fit_predict_regime(train_df, train_df, cfg.regime_vars, cfg.regime_k, cfg.regime_pca_components)
        validate_df, _ = fit_predict_regime(train_df, validate_df, cfg.regime_vars, cfg.regime_k, cfg.regime_pca_components)
        test_df, _ = fit_predict_regime(train_df, test_df, cfg.regime_vars, cfg.regime_k, cfg.regime_pca_components)

        regime_dummy_cols = [c for c in train_df.columns if c.startswith("regime_")]
        feat_cols = base_features + regime_dummy_cols
        
        for df_part in (train_df, validate_df, test_df):
            for col in engineered_cols_static + external_extra_features:
                if col not in df_part.columns: continue
                transform_func = lambda s: s.fillna(s.median(skipna=True)) if s.notna().any() else s.fillna(0.0)
                df_part[col] = df_part[col].fillna(df_part.groupby(date_col)[col].transform(transform_func))
            df_part[engineered_cols_static + external_extra_features] = df_part[engineered_cols_static + external_extra_features].fillna(0.0)

        # --- Standardize base numeric features on training only for linear model ---
        print(f"[{ts()}] Fitting StandardScaler on training features…")
        scaler = StandardScaler().fit(train_df[feat_cols])
        # MEMORY OPTIMIZATION: Transform columns in-place
        train_df[feat_cols] = scaler.transform(train_df[feat_cols])
        validate_df[feat_cols] = scaler.transform(validate_df[feat_cols])
        test_df[feat_cols] = scaler.transform(test_df[feat_cols])

        # --- arrays ---
        X_train = train_df[feat_cols].values
        Y_train = train_df[ret_var].values
        X_val   = validate_df[feat_cols].values
        Y_val   = validate_df[ret_var].values
        X_test  = test_df[feat_cols].values
        Y_test  = test_df[ret_var].values

        print(f"Y_train: min={np.min(Y_train)}, max={np.max(Y_train)}, mean={np.mean(Y_train)}, std={np.std(Y_train)}")
        print(f"Y_test: min={np.min(Y_test)}, max={np.max(Y_test)}, mean={np.mean(Y_test)}, std={np.std(Y_test)}")

        keep_cols = ["year", "month", "ret_eom", id_col, ret_var, date_col]
        keep_cols_present = [c for c in keep_cols if c in test_df.columns]
        reg_pred = test_df[keep_cols_present].copy()

        # RIDGE
        print(f"[{ts()}] RIDGE: tuning...")
        lambdas = np.arange(-1, 8.1, 0.2)
        val_mse_ridge = np.zeros(len(lambdas))
        t_ridge0 = time.perf_counter()
        Y_train_dm = Y_train - np.mean(Y_train)
        for ind, i in enumerate(lambdas):
            reg = Ridge(alpha=((10**i) * 0.5), fit_intercept=False, solver="svd")
            reg.fit(X_train, Y_train_dm)
            preds = reg.predict(X_val) + np.mean(Y_train)
            val_mse_ridge[ind] = mean_squared_error(Y_val, preds)
        best_lambda = lambdas[np.argmin(val_mse_ridge)]
        reg = Ridge(alpha=((10**best_lambda) * 0.5), fit_intercept=False, solver="svd")
        reg.fit(X_train, Y_train_dm)
        preds_ridge_test = reg.predict(X_test) + np.mean(Y_train)
        reg_pred["ridge"] = preds_ridge_test
        t_ridge1 = time.perf_counter()
        ridge_mse_test = mean_squared_error(Y_test, preds_ridge_test)
        ridge_mse_val  = np.min(val_mse_ridge)
        print(f"[{ts()}] RIDGE: best alpha=0.5*1e{best_lambda:.1f} | Val MSE={ridge_mse_val:.8f} | Test MSE={ridge_mse_test:.8f} | time={t_ridge1 - t_ridge0:,.2f}s")

        # XGBOOST
        print(f"[{ts()}] XGBoost: fitting...")
        xgb = XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                           reg_lambda=1.0, tree_method="hist", n_jobs=-1, random_state=2025, eval_metric="rmse", early_stopping_rounds=100)
        t_xgb0 = time.perf_counter()
        xgb.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=False)
        t_xgb1 = time.perf_counter()
        preds_xgb_test = xgb.predict(X_test)
        xgb_mse_val  = mean_squared_error(Y_val, xgb.predict(X_val))
        xgb_mse_test = mean_squared_error(Y_test, preds_xgb_test)
        reg_pred["xgb"] = preds_xgb_test
        print(f"[{ts()}] XGBoost: best_iteration={xgb.best_iteration} | Val MSE={xgb_mse_val:.8f} | Test MSE={xgb_mse_test:.8f} | time={t_xgb1 - t_xgb0:,.2f}s")

        # CATBOOST
        print(f"[{ts()}] CatBoost: fitting...")
        cat = CatBoostRegressor(loss_function="RMSE", depth=6, learning_rate=0.05, l2_leaf_reg=3.0, iterations=2000,
                                random_seed=2025, subsample=0.8, rsm=0.8, od_type="Iter", od_wait=100, verbose=False)
        t_cat0 = time.perf_counter()
        cat.fit(Pool(X_train, Y_train), eval_set=Pool(X_val, Y_val), use_best_model=True)
        t_cat1 = time.perf_counter()
        preds_cat_test = cat.predict(X_test)
        cat_mse_val  = mean_squared_error(Y_val, cat.predict(X_val))
        cat_mse_test = mean_squared_error(Y_test, preds_cat_test)
        reg_pred["cat"] = preds_cat_test
        print(f"[{ts()}] CatBoost: best_iteration={cat.get_best_iteration()} | Val MSE={cat_mse_val:.8f} | Test MSE={cat_mse_test:.8f} | time={t_cat1 - t_cat0:,.2f}s")

        # BLENDING
        mse_vals = {"ridge": float(ridge_mse_val), "xgb": float(xgb_mse_val), "cat": float(cat_mse_val)}
        eps = 1e-12
        weights = {k: 1.0 / (v + eps) for k, v in mse_vals.items()}
        w_sum = sum(weights.values())
        weights = {k: v / w_sum for k, v in weights.items()}
        print(f"[{ts()}] Blend Weights: ridge={weights['ridge']:.3f}, xgb={weights['xgb']:.3f}, cat={weights['cat']:.3f}")
        blend_test = (weights["ridge"] * reg_pred["ridge"].values + weights["xgb"] * reg_pred["xgb"].values + weights["cat"] * reg_pred["cat"].values)
        reg_pred["blend"] = blend_test
        blend_mse_test = mean_squared_error(Y_test, blend_test)
        print(f"[{ts()}] BLEND: Test MSE={blend_mse_test:.8f}")

        reg_pred["iter"] = loop_iter
        reg_pred["train_start"], reg_pred["train_end"] = c0.date(), (c1 - pd.Timedelta(days=1)).date()
        reg_pred["val_end"], reg_pred["test_end"] = (c2 - pd.Timedelta(days=1)).date(), (c3 - pd.Timedelta(days=1)).date()
        
        try:
            _append_preds(reg_pred, out_path)
            print(f"[{ts()}] Appended {len(reg_pred):,} rows to {out_path}")
        except Exception as e:
            print(f"[{ts()}] ERROR appending iteration {loop_iter} to {out_path}: {e}")

        # --- MEMORY OPTIMIZATION: Clean up at end of loop ---
        del train_df, validate_df, test_df, X_train, Y_train, X_val, Y_val, X_test, Y_test, reg_pred
        gc.collect()
        
        counter += 1

    # --- OOS R² (vs zero) ---
    print(hrule())
    print(f"[{ts()}] Reading final predictions from {out_path} for OOS R2 calculation.")
    
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        pred_out = pd.read_csv(out_path)
        print(f"[{ts()}] Computing overall OOS R2 (relative to zero):")
        yreal = pred_out[ret_var].values
        denom = np.sum(np.square(yreal))
        print(f"[{ts()}] Denominator sum(y^2) = {denom:.8f}")
        for model_name in ["ridge", "xgb", "cat", "blend"]:
            if model_name not in pred_out.columns:
                print(f"[{ts()}] {model_name}: no predictions found.")
                continue
            ypred = pred_out[model_name].values
            sse = np.sum(np.square((yreal - ypred)))
            r2 = 1 - sse / denom if denom != 0 else np.nan
            print(f"[{ts()}] {model_name.upper():6s} | SSE={sse:.8f} | R2_zero_bench={r2:.8f}")
    else:
        print(f"[{ts()}] {out_path} not found or is empty. Cannot calculate OOS R².")

    t1 = time.perf_counter()
    print(hrule())
    print(f"[{ts()}] END | Total runtime: {t1 - t0:,.2f}s")

