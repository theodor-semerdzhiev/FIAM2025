import datetime
import time
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost.callback import EarlyStopping

# --- ADDED: plotting backend for headless environments ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# --------------------------------------------------------

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
    df["pp_vol_w"]      = vol.values
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
    work_dir = "/Users/loading.../Documents/Projects/FIAM2025"
    print(f"[{ts()}] Working directory: {work_dir}")

    # --- data in ---
    file_path = os.path.join(work_dir, "data/ret_sample.parquet")
    print(f"[{ts()}] Reading returns data (parquet): {file_path}")
    raw = pd.read_parquet(file_path, engine="pyarrow")
    # NEW: force float32 to cut memory
    _float64_cols = raw.select_dtypes(include=["float64"]).columns
    if len(_float64_cols) > 0:
        raw[_float64_cols] = raw[_float64_cols].astype("float32")

    print(f"[{ts()}] raw shape: {raw.shape}; columns: {len(raw.columns)}")

    file_path = os.path.join(work_dir, "data/factor_char_list.csv")
    print(f"[{ts()}] Reading factor list: {file_path}")
    stock_vars = list(pd.read_csv(file_path)["variable"].values)
    print(f"[{ts()}] #factors: {len(stock_vars)}")

    # --- target & checks ---
    ret_var = "stock_ret"
    id_col = "id"
    date_col = "date"

    if ret_var not in raw.columns:
        raise KeyError(f"Column '{ret_var}' not found in raw data. Available (first 20): {list(raw.columns)[:20]}")
    if date_col not in raw.columns:
        raise KeyError("Column 'date' not found. The monthly grouping requires 'date'.")
    if id_col not in raw.columns:
        # try a common fallback
        id_col = [c for c in ["permno", "ticker", "cusip", "gvkey"] if c in raw.columns][0]
        print(f"[{ts()}] Using '{id_col}' as security id column.")

    print(f"[{ts()}] Parsing 'date' column once…")
    print(raw[date_col].head(3))
    raw[date_col] = parse_date_col(raw[date_col])

    missing_before = raw[ret_var].isna().sum()
    print(f"[{ts()}] Rows with missing {ret_var} before filter: {missing_before:,}")
    data0 = raw[raw[ret_var].notna()].copy()
    print(f"[{ts()}] shape after drop NA in {ret_var}: {data0.shape}")
    if data0.empty:
        raise RuntimeError("Dataset is empty after dropping missing targets.")
    print(f"[{ts()}] Date coverage: {data0[date_col].min().date()} → {data0[date_col].max().date()}")


    # # === NEW: Merge per-id, per-date MGMT_PRED feature (same spot as risk/similarity) ===
    # mgmt_path = os.path.join(work_dir, "data/mgmt_pred_feature.parquet")
    # print(f"[{ts()}] Reading mgmt_pred features: {mgmt_path}")

    # # Expecting columns: id, date, mgmt_pred (gvkey may be present but not required)
    # mgmt = pd.read_parquet(mgmt_path, engine="pyarrow")

    # # Keep needed cols
    # need_cols = ["id", "date", "mgmt_pred"]
    # missing = [c for c in need_cols if c not in mgmt.columns]
    # if missing:
    #     raise KeyError(f"mgmt_pred_feature.parquet missing columns: {missing}")

    # mgmt = mgmt[need_cols].copy()

    # # Normalize types to match main
    # mgmt["date"] = parse_date_col(mgmt["date"])
    # mgmt["id"] = mgmt["id"].astype(str)
    # mgmt["mgmt_pred"] = pd.to_numeric(mgmt["mgmt_pred"], errors="coerce")

    # # Deduplicate last-by-time
    # mgmt = mgmt.sort_values(["id", "date"]).drop_duplicates(["id", "date"], keep="last")

    # # Ensure main keys are aligned
    # data0["id"] = data0["id"].astype(str)

    # # Merge m:1 on (id,date)
    # pre_rows = len(data0)
    # data0 = data0.merge(mgmt, on=["id", "date"], how="left", validate="m:1")
    # post_rows = len(data0)
    # assert pre_rows == post_rows, "Row count changed unexpectedly after mgmt_pred merge."

    # matched_mgmt = data0["mgmt_pred"].notna().sum()
    # print(f"[{ts()}] mgmt_pred merged: matched {matched_mgmt:,} of {len(data0):,} rows.")
    # # === END mgmt_pred merge ===


    # # === Merge per-id, per-date risk_score only ===
    # aux_path = os.path.join(work_dir, "data/text_based_risk_scores.csv")  # adjust to your file name

    # print(f"[{ts()}] Reading risk_score file: {aux_path}")
    # aux = pd.read_csv(aux_path, usecols=["id", "date_dt", "risk_score"])

    # # Normalize date
    # aux["date_dt"] = parse_date_col(aux["date_dt"])
    # aux = aux.rename(columns={"date_dt": "date"})
    # aux["id"] = aux["id"].astype(str)
    # data0["id"] = data0["id"].astype(str)

    # # Deduplicate if necessary
    # aux = aux.sort_values(["id", "date"]).drop_duplicates(["id", "date"], keep="last")

    # # Merge risk_score into main data
    # pre_rows = len(data0)
    # data0 = data0.merge(aux, on=["id", "date"], how="left", validate="m:1")
    # post_rows = len(data0)
    # assert pre_rows == post_rows, "Row count changed unexpectedly after merge."

    # # Diagnostics
    # matched = data0["risk_score"].notna().sum()
    # print(f"[{ts()}] risk_score merged: matched {matched:,} of {len(data0):,} rows.")

    # # === NEW: Merge per-id, per-date SIMILARITY features ===
    # sim_path = os.path.join(work_dir, "data/similarity_scores.csv")
    # print(f"[{ts()}] Reading similarity features: {sim_path}")

    # sim_usecols = [
    #     "id","date",
    #     "similarity_score","similarity_confidence","similarity_matches",
    #     "similarity_volatility","similarity_strength","similarity_rank_pct",
    #     "similarity_vs_median","high_confidence_similarity"
    # ]
    # sim = pd.read_csv(sim_path, usecols=sim_usecols)

    # # Normalize types
    # sim["date"] = parse_date_col(sim["date"])
    # sim["id"] = sim["id"].astype(str)

    # # (Optional but robust) coerce numerics
    # for c in sim.columns:
    #     if c not in ("id","date"):
    #         sim[c] = pd.to_numeric(sim[c], errors="coerce")

    # # Deduplicate (last wins)
    # sim = sim.sort_values(["id","date"]).drop_duplicates(["id","date"], keep="last")

    # # Merge
    # pre_rows = len(data0)
    # data0 = data0.merge(sim, on=["id","date"], how="left", validate="m:1")
    # post_rows = len(data0)
    # assert pre_rows == post_rows, "Row count changed unexpectedly after similarity merge."

    # # Diagnostics
    # sim_cols = [c for c in sim_usecols if c not in ("id","date")]
    # sim_match_any = data0[sim_cols].notna().any(axis=1).sum()
    # sim_match_main = data0["similarity_score"].notna().sum()
    # print(f"[{ts()}] similarity_* merged: any-feature matched {sim_match_any:,} rows; "
    #     f"similarity_score matched {sim_match_main:,} rows.")



    # missing_factors = [v for v in stock_vars if v not in data0.columns]
    # if missing_factors:
    #     raise KeyError(f"Missing factor(s) in data: {missing_factors[:10]}{' …' if len(missing_factors)>10 else ''}")

    # # --- monthly rank-scaling to [-1,1] ---
    # print(f"[{ts()}] Grouping cross-sections by '{date_col}' and rank-scaling to [-1,1] per month")
    # monthly = data0.groupby(date_col, sort=True)
    # chunks = []
    # for m_i, (date_val, monthly_raw) in enumerate(monthly, start=1):
    #     group = monthly_raw.copy()
    #     zeroed_vars_this_month = 0
    #     for var in stock_vars:
    #         med = group[var].median(skipna=True)
    #         group[var] = group[var].fillna(med)
    #         group[var] = group[var].rank(method="dense") - 1
    #         gmax = group[var].max()
    #         if pd.isna(gmax) or gmax <= 0:
    #             group[var] = 0
    #             zeroed_vars_this_month += 1
    #         else:
    #             group[var] = (group[var] / gmax) * 2 - 1
    #     chunks.append(group)
    #     if m_i <= 3 or m_i % 12 == 0:
    #         print(f"[{ts()}] {date_val.date()} | zeroed vars={zeroed_vars_this_month}")
    # data = pd.concat(chunks, ignore_index=True)


    # --- leak-safe rolling features computed once (do *not* use future info) ---
    cfg = FeatureConfig(
        regime_vars=stock_vars,               # you can pass a curated subset here later
        regime_k=4,
        regime_pca_components=min(6, len(stock_vars)),
        momentum_vars=stock_vars[:12],        # default: first dozen as a proxy for 'important'
        momentum_window=6,
        path_window=6,
    )

    # print(hrule("-"))
    # print(f"[{ts()}] Computing leak-safe rolling PRICE PATH features (window={cfg.path_window})…")
    # data = add_price_path_features(data, id_col=id_col, date_col=date_col, ret_col=ret_var, path_window=cfg.path_window)

    # print(f"[{ts()}] Computing FACTOR MOMENTUM features (vars={len(cfg.momentum_vars)}, window={cfg.momentum_window})…")
    # data = add_factor_momentum(data, date_col=date_col, ret_col=ret_var, vars_=cfg.momentum_vars, window=cfg.momentum_window)

    # print(f"[{ts()}] Computing META features (v2 transforms)…")
    # data = add_meta_features_v2(data, date_col=date_col, id_col=id_col)

    ####Comment this section to run without the saved file ###############
    rankscaled_path = os.path.join(work_dir, "data/rank_scaled.parquet")
    # print(f"[{ts()}] Saving rank-scaled data to: {rankscaled_path}")
    # data.to_parquet(rankscaled_path, engine="pyarrow", index=False)

    data = pd.read_parquet(rankscaled_path, engine="pyarrow")
    # NEW: force float32 to cut memory
    _float64_cols_cache = data.select_dtypes(include=["float64"]).columns
    if len(_float64_cols_cache) > 0:
        data[_float64_cols_cache] = data[_float64_cols_cache].astype("float32")

    print(f"[{ts()}] Loaded cached rank-scaled data, shape={data.shape}")
    ####End of comment section to run without the saved file ###############




    # Collect engineered columns here; regime added inside the loop because it must be fit on train only
    engineered_cols_static = [
        "price_path", "pp_cumret_w", "pp_vol_w", "pp_downside_w", "pp_signchg_w", "factor_momentum"
    ]

    # NEW: declare extra (non-ranked) external features that should feed the models
    external_extra_features = [
        "risk_score",
        "similarity_score","similarity_confidence","similarity_matches",
        "similarity_volatility","similarity_strength","similarity_rank_pct",
        "similarity_vs_median","high_confidence_similarity",
        "mgmt_pred"
    ]



    # =========================
    # walk-forward loop
    # =========================
    print(hrule())
    print(f"[{ts()}] BEGIN walk-forward loop (expanding train, yearly retrain)")

    starting = pd.to_datetime("2005-01-01")
    counter = 0
    loop_iter = 0
    pred_frames = []

    # master list of base vars + engineered features (regime dummies added per-iter)
    # stock_vars are already rank-scaled; engineered & external features stay in native scale
    base_features = stock_vars + engineered_cols_static + external_extra_features



    out_path = os.path.join(work_dir, "output.csv")

    def _append_preds(df: pd.DataFrame, path: str):
        """Append predictions to CSV with header only if file is new/empty."""
        # Ensure consistent column order across iterations
        desired = ["year","month","ret_eom", id_col, ret_var, date_col, "ridge","xgb","cat","blend",
                "iter","train_start","train_end","val_end","test_end"]
        for c in desired:
            if c not in df.columns:
                df[c] = np.nan
        df = df[desired]

        write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
        df.to_csv(path, mode="a", header=write_header, index=False)

    while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime("2026-01-01"):

        loop_iter += 1
        # ✅ Skip until the 7th iteration OR after a certain year
        if loop_iter < 11:
            counter += 1
            continue

        c0 = starting
        c1 = starting + pd.DateOffset(years=8 + counter)   # training upper bound
        c2 = starting + pd.DateOffset(years=10 + counter)  # validation upper bound
        c3 = starting + pd.DateOffset(years=11 + counter)  # testing upper bound

        print(hrule())
        print(f"[{ts()}] ITER {loop_iter} | counter={counter}")
        print(f"[{ts()}] Train:    [{c0.date()} → {(c1 - pd.Timedelta(days=1)).date()} ]")
        print(f"[{ts()}] Validate: [ {c1.date()} → {(c2 - pd.Timedelta(days=1)).date()} ]")
        print(f"[{ts()}] Test:     [ {c2.date()} → {(c3 - pd.Timedelta(days=1)).date()} ]")

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
        train_df_reg, artifacts = fit_predict_regime(train_df, train_df, cfg.regime_vars, cfg.regime_k, cfg.regime_pca_components)
        validate_df_reg, _ = fit_predict_regime(train_df, validate_df, cfg.regime_vars, cfg.regime_k, cfg.regime_pca_components)
        test_df_reg, _ = fit_predict_regime(train_df, test_df, cfg.regime_vars, cfg.regime_k, cfg.regime_pca_components)

        regime_dummy_cols = [c for c in train_df_reg.columns if c.startswith("regime_")]

        # --- Feature matrix assembly ---
        feat_cols = base_features + regime_dummy_cols

        # Some engineered features may be NA early in the sample; fill with cross-sectional median per date
        # replace this inside the loop over df_part:
        # df_part[col] = df_part[col].fillna(df_part.groupby(date_col)[col].transform(lambda s: s.fillna(s.median())))

        for df_part in (train_df_reg, validate_df_reg, test_df_reg):
            # Fill engineered features per-date
            for col in engineered_cols_static:
                if col not in df_part.columns: 
                    continue
                df_part[col] = df_part[col].fillna(
                    df_part.groupby(date_col)[col].transform(
                        lambda s: s.fillna(s.median(skipna=True)) if s.notna().any() else s.fillna(0.0)
                    )
                )
            # NEW: Fill external similarity/risk features per-date
            for col in external_extra_features:
                if col not in df_part.columns: 
                    continue
                df_part[col] = df_part[col].fillna(
                    df_part.groupby(date_col)[col].transform(
                        lambda s: s.fillna(s.median(skipna=True)) if s.notna().any() else s.fillna(0.0)
                    )
                )
            # last resort
            df_part[engineered_cols_static] = df_part[engineered_cols_static].fillna(0.0)
            df_part[external_extra_features] = df_part[external_extra_features].fillna(0.0)



        # --- Standardize base numeric features on training only for linear model ---
        print(f"[{ts()}] Fitting StandardScaler on training features…")
        scaler = StandardScaler().fit(train_df_reg[feat_cols])
        train_std = train_df_reg.copy(); train_std[feat_cols] = scaler.transform(train_df_reg[feat_cols]).astype(np.float32)  # NEW: force float32 to cut memory
        val_std   = validate_df_reg.copy(); val_std[feat_cols]   = scaler.transform(validate_df_reg[feat_cols]).astype(np.float32)  # NEW: force float32 to cut memory
        test_std  = test_df_reg.copy();  test_std[feat_cols]  = scaler.transform(test_df_reg[feat_cols]).astype(np.float32)  # NEW: force float32 to cut memory

        # --- arrays ---
        X_train = train_std[feat_cols].to_numpy(dtype=np.float32, copy=False)  # NEW: force float32 to cut memory
        Y_train = train_std[ret_var].to_numpy(dtype=np.float32, copy=False)    # NEW: force float32 to cut memory
        X_val   = val_std[feat_cols].to_numpy(dtype=np.float32, copy=False)    # NEW: force float32 to cut memory
        Y_val   = val_std[ret_var].to_numpy(dtype=np.float32, copy=False)      # NEW: force float32 to cut memory
        X_test  = test_std[feat_cols].to_numpy(dtype=np.float32, copy=False)   # NEW: force float32 to cut memory
        Y_test  = test_std[ret_var].to_numpy(dtype=np.float32, copy=False)     # NEW: force float32 to cut memory

        # ...existing code...
        print(f"Y_train: min={np.min(Y_train)}, max={np.max(Y_train)}, mean={np.mean(Y_train)}, std={np.std(Y_train)}")
        print(f"Y_test: min={np.min(Y_test)}, max={np.max(Y_test)}, mean={np.mean(Y_test)}, std={np.std(Y_test)}")
        # ...existing code...

        # --- prediction frame skeleton ---
        keep_cols = ["year", "month", "ret_eom", id_col, ret_var, date_col]
        keep_cols_present = [c for c in keep_cols if c in test_std.columns]
        reg_pred = test_std[keep_cols_present].copy()

        # =========================
        # RIDGE
        # =========================
        print(f"[{ts()}] RIDGE: tuning alpha over 10^[-1, ..., 8] (x0.5) step=0.2")
        lambdas = np.arange(-1, 8.1, 0.2)
        val_mse_ridge = np.zeros(len(lambdas), dtype=np.float32)  # NEW: force float32 to cut memory
        t_ridge0 = time.perf_counter()
        Y_train_dm = Y_train - np.mean(Y_train)
        for ind, i in enumerate(lambdas):
            reg = Ridge(alpha=((10**i) * 0.5), fit_intercept=False, solver="svd")
            reg.fit(X_train, Y_train_dm)
            preds = reg.predict(X_val) + np.mean(Y_train)
            val_mse_ridge[ind] = mean_squared_error(Y_val, preds)
            if ind % 10 == 0 or ind == len(lambdas) - 1:
                print(f"[{ts()}]   alpha=0.5*1e{i: .1f} | val MSE={val_mse_ridge[ind]:.8f}")
        best_lambda = lambdas[np.argmin(val_mse_ridge)]
        reg = Ridge(alpha=((10**best_lambda) * 0.5), fit_intercept=False, solver="svd")
        reg.fit(X_train, Y_train_dm)
        preds_ridge_test = reg.predict(X_test) + np.mean(Y_train)
        print(f"Preds: min={np.min(preds_ridge_test)}, max={np.max(preds_ridge_test)}, mean={np.mean(preds_ridge_test)}, std={np.std(preds_ridge_test)}")
        reg_pred["ridge"] = preds_ridge_test
        t_ridge1 = time.perf_counter()
        ridge_mse_test = mean_squared_error(Y_test, preds_ridge_test)
        ridge_mse_val  = np.min(val_mse_ridge)
        print(f"[{ts()}] RIDGE: best alpha=0.5*1e{best_lambda:.1f} | Val MSE={ridge_mse_val:.8f} | Test MSE={ridge_mse_test:.8f} | time={t_ridge1 - t_ridge0:,.2f}s")

        # =========================
        # XGBOOST (GBDT)
        # =========================
        print(f"[{ts()}] XGBoost: fitting with early stopping…")
        xgb = XGBRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            min_child_weight=1.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=2025,
            eval_metric="rmse",
            early_stopping_rounds=100
        )
        t_xgb0 = time.perf_counter()
        xgb.fit(
            X_train, Y_train,
            eval_set=[(X_val, Y_val)],
            verbose=False,
        )

        t_xgb1 = time.perf_counter()
        preds_xgb_val  = xgb.predict(X_val)
        preds_xgb_test = xgb.predict(X_test)
        xgb_mse_val  = mean_squared_error(Y_val, preds_xgb_val)
        xgb_mse_test = mean_squared_error(Y_test, preds_xgb_test)
        reg_pred["xgb"] = preds_xgb_test
        print(f"[{ts()}] XGBoost: best_iteration={xgb.best_iteration} | Val MSE={xgb_mse_val:.8f} | Test MSE={xgb_mse_test:.8f} | time={t_xgb1 - t_xgb0:,.2f}s")

        # =========================
        # ADDED: XGBoost Feature Importance (per-iteration Top-20)
        # =========================
        try:
            # Ensure output dirs exist
            fi_dir = os.path.join(work_dir, "feature_importance")
            os.makedirs(fi_dir, exist_ok=True)

            # Map XGBoost's f{idx} back to your column names (order = feat_cols)
            fmap = {f"f{i}": feat_cols[i] for i in range(len(feat_cols))}

            # Choose importance metric: 'gain' (avg gain), alternatives: 'weight','cover','total_gain','total_cover'
            raw_score = xgb.get_booster().get_score(importance_type="gain")

            # Replace keys with real feature names and build a Series
            if len(raw_score) == 0:
                print(f"[{ts()}] WARNING: XGB returned empty importance; skipping plot.")
            else:
                imp = pd.Series({fmap.get(k, k): v for k, v in raw_score.items()}, dtype="float64")
                imp = imp.sort_values(ascending=False)
                topk = imp.head(20)

                # Save CSV per iteration
                csv_path = os.path.join(fi_dir, f"xgb_importance_iter_{loop_iter:03d}.csv")
                imp.reset_index().rename(columns={"index": "feature", 0: "gain"}).to_csv(csv_path, index=False)
                print(f"[{ts()}] Saved XGB importance CSV: {csv_path}")

                # Plot Top-20 (horizontal bar), highest on top
                plt.figure(figsize=(9, 7))
                topk.iloc[::-1].plot(kind="barh")
                plt.xlabel("Gain")
                plt.ylabel("Feature")
                plt.title(f"XGBoost Feature Importance (Top 20) — Iter {loop_iter}")
                plt.tight_layout()
                png_path = os.path.join(fi_dir, f"xgb_importance_iter_{loop_iter:03d}.png")
                plt.savefig(png_path, dpi=150)
                plt.close()
                print(f"[{ts()}] Saved XGB importance plot: {png_path}")
        except Exception as _fi_err:
            print(f"[{ts()}] WARNING: failed to compute/plot XGB feature importance in iter {loop_iter}: {_fi_err}")
        # =========================
        # END ADDED BLOCK
        # =========================

        # =========================
        # CATBOOST (GBDT)
        # =========================
        print(f"[{ts()}] CatBoost: fitting with early stopping…")
        train_pool = Pool(X_train, Y_train)
        val_pool   = Pool(X_val,   Y_val)
        cat = CatBoostRegressor(
            loss_function="RMSE",
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            iterations=2000,
            random_seed=2025,
            subsample=0.8,
            rsm=0.8,
            od_type="Iter",
            od_wait=100,
            verbose=False
        )
        t_cat0 = time.perf_counter()
        cat.fit(train_pool, eval_set=val_pool, use_best_model=True)
        t_cat1 = time.perf_counter()
        preds_cat_val  = cat.predict(X_val)
        preds_cat_test = cat.predict(X_test)
        cat_mse_val  = mean_squared_error(Y_val, preds_cat_val)
        cat_mse_test = mean_squared_error(Y_test, preds_cat_test)
        reg_pred["cat"] = preds_cat_test
        print(f"[{ts()}] CatBoost: best_iteration={cat.get_best_iteration()} | Val MSE={cat_mse_val:.8f} | Test MSE={cat_mse_test:.8f} | time={t_cat1 - t_cat0:,.2f}s")

        # =========================
        # BLENDING (inverse val-MSE weights)
        # =========================
        mse_vals = {
            "ridge": float(ridge_mse_val),
            "xgb":   float(xgb_mse_val),
            "cat":   float(cat_mse_val)
        }
        eps = 1e-12
        weights = {k: 1.0 / (v + eps) for k, v in mse_vals.items()}
        w_sum = sum(weights.values())
        weights = {k: v / w_sum for k, v in weights.items()}
        print(f"[{ts()}] Blend Weights: ridge={weights['ridge']:.3f}, xgb={weights['xgb']:.3f}, cat={weights['cat']:.3f}")

        blend_test = (
            weights["ridge"] * reg_pred["ridge"].values +
            weights["xgb"]   * reg_pred["xgb"].values +
            weights["cat"]   * reg_pred["cat"].values
        )
        reg_pred["blend"] = blend_test
        blend_mse_test = mean_squared_error(Y_test, blend_test)
        print(f"[{ts()}] BLEND: Test MSE={blend_mse_test:.8f}")

        # ADD THESE LINES after the IC calculation:
        sign_correct = (np.sign(blend_test) == np.sign(Y_test)).mean()
        ic_spearman = pd.Series(blend_test).corr(pd.Series(Y_test), method='spearman')
        print(f"[{ts()}] Sign Accuracy: {sign_correct:.3f} | Spearman IC: {ic_spearman:.4f}")

        # R² calculations per iteration
        denom = np.sum(np.square(Y_test))
        r2_ridge = 1 - np.sum(np.square(Y_test - preds_ridge_test)) / denom if denom != 0 else np.nan
        r2_xgb = 1 - np.sum(np.square(Y_test - preds_xgb_test)) / denom if denom != 0 else np.nan
        r2_cat = 1 - np.sum(np.square(Y_test - preds_cat_test)) / denom if denom != 0 else np.nan
        r2_blend = 1 - np.sum(np.square(Y_test - blend_test)) / denom if denom != 0 else np.nan

        print(f"[{ts()}] R² (vs zero) -> Ridge: {r2_ridge:.6f} | XGB: {r2_xgb:.6f} | Cat: {r2_cat:.6f} | Blend: {r2_blend:.6f}")

        reg_pred["iter"] = loop_iter
        reg_pred["train_start"] = c0.date()
        reg_pred["train_end"]   = (c1 - pd.Timedelta(days=1)).date()
        reg_pred["val_end"]     = (c2 - pd.Timedelta(days=1)).date()
        reg_pred["test_end"]    = (c3 - pd.Timedelta(days=1)).date()

        # ---- NEW: write this iteration's predictions immediately ----
        try:
            _append_preds(reg_pred, out_path)
            print(f"[{ts()}] Appended {len(reg_pred):,} rows to {out_path}")
        except Exception as e:
            print(f"[{ts()}] ERROR appending iteration {loop_iter} to {out_path}: {e}")
        # -------------------------------------------------------------

        pred_frames.append(reg_pred)
        counter += 1

    # --- concat all predictions ---
    pred_out = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()

    # --- write CSV ---
    out_path = os.path.join(work_dir, "output.csv")
    print(hrule())
    print(f"[{ts()}] Writing predictions to: {out_path}")
    if pred_out.empty:
        print(f"[{ts()}] WARNING: No predictions produced.")
        minimal_cols = [date_col, ret_var, "ridge", "xgb", "cat", "blend"]
        pd.DataFrame(columns=minimal_cols).to_csv(out_path, index=False)
        print(f"[{ts()}] Wrote CSV with shape: (0, {len(minimal_cols)})")
    else:
        pred_out.to_csv(out_path, index=False)
        print(f"[{ts()}] Wrote CSV with shape: {pred_out.shape}")

    # --- OOS R² (vs zero) ---
    print(hrule())
    if pred_out.empty or ret_var not in pred_out.columns:
        print(f"[{ts()}] No predictions or '{ret_var}' missing; skipping OOS R².")
    else:
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
            print(f"[{ts()}] {model_name.UPPER():6s} | SSE={sse:.8f} | R2_zero_bench={r2:.8f}")

    t1 = time.perf_counter()
    print(hrule())
    print(f"[{ts()}] END | Total runtime: {t1 - t0:,.2f}s")
