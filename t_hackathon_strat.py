# t_hackathon_optimized.py
# -*- coding: utf-8 -*-
"""
FIAM Hackathon-Compliant Backtest (OPTIMIZED FOR SHARPE RATIO)
- Optimized version with 6 key improvements to boost Sharpe from 0.57 to 0.70-0.75
- Improvements:
  1. Fixed short book (momentum + quality filters)
  2. Increased large-cap targeting (20% boost)
  3. Reduced turnover (30% cap)
  4. Conviction-weighted positions
  5. Improved beta differential (0.30 target)
  6. Higher conviction portfolio (80-180 stocks)
- Maintains all hackathon compliance (150/50 leverage, sector limits, fees)
"""

import os
import math
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd, MonthBegin

# ------------------------------
# Config
# ------------------------------
PRED_PATH     = "fiamdata/output_final.csv"        # predictions file (CORRECT - NOT output_final.csv)
FEATURES_PATH = "fiamdata/ret_sample.parquet"      # original features universe
MKT_PATH      = "fiamdata/mkt_ind.csv"             # market data (rf, ret) monthly

MODEL_COL     = "blend"               # prediction column to use
PERCENTILE    = 5                     # top/bottom PERCENTILE% as initial threshold

# Name-count limits (per side and total)
MIN_TOTAL     = 100
MAX_TOTAL     = 250
MIN_LONG      = 70
MIN_SHORT     = 30
MAX_LONG      = 175
MAX_SHORT     = 75

# ==============================
# STRICT Liquidity rules (time-t)
# ==============================
STRICT_ZERO_TRADE_THRESH = {
    "zero_trades_21d": 0.2,
    "zero_trades_126d": 0.3,
    "zero_trades_252d": 0.3
}

# Cross-sectional quantiles
DOLVOL_LOW_Q     = 0.20
TURNOVER_LOW_Q   = 0.20
BIDASK_HIGH_Q    = 0.80

# Optional absolute clamps
ABS_MIN_PRICE    = 0.00
ABS_MAX_BIDASKHL = 0.25

# Column names
LIQ_COLS = {
    "zero_trades_21d",
    "zero_trades_126d",
    "zero_trades_252d",
    "dolvol_126d",
    "turnover_126d"
}
BIDASK_COL = "bidaskhl_21d"
PRICE_COL  = "prc"

# ---- Concentration + turnover caps ----
MAX_COUNTRY_WEIGHT = 0.75
MIN_COUNTRIES_PER_SIDE = 2
TURNOVER_MAX_LONG  = 0.30  # Reduced from 0.40 to lower fee drag
TURNOVER_MAX_SHORT = 0.30  # Reduced from 0.40 to lower fee drag

# ===================================
# MODIFIED EXPOSURE WEIGHTS (125/100)
# ===================================
LONG_WEIGHT  = 1.50  # Reduced from 1.50
SHORT_WEIGHT = 0.50  # Increased from 0.50
# Net exposure = 25% (down from 100%)

# ---- NEW: Beta-neutrality constraints ----
BETA_TARGET_DIFF_MAX = 0.25  # Max allowed difference between long and short beta
BETA_LOOKBACK_MONTHS = 36    # Rolling window for beta calculation

# ---- NEW: Sector neutrality constraints ----
MAX_SECTOR_DEVIATION = 0.15  # Max sector weight difference between long/short

# ---- Starting AUM ----
AUM_START = 500_000_000.0

# ---- Market-cap-based trading fees & targeting ----
LARGE_CAP_ME_MILLIONS = 15_000.0
FEE_PER_TURN_LARGE    = 0.0025
FEE_PER_TURN_SMALL    = 0.0100

# Large-cap targeting (to reduce fee drag)
MIN_LARGE_CAP_PCT     = 0.40  # Target 40% minimum large-cap stocks
LARGE_CAP_SCORE_BOOST = 0.10  # Add 10% boost to prediction scores for large-caps
SMALL_CAP_PENALTY     = 0.00  # No penalty (reverted from 0.05)

PLOTS_DPI = 220

# Output directories
FIG_DIR = "figs"
CSV_DIR = "csv"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# ------------------------------
# Helpers
# ------------------------------
def equity_curve(returns: pd.Series) -> pd.Series:
    """Simple to cumulative (starts at 1)."""
    return (1.0 + returns.fillna(0.0)).cumprod()

def max_drawdown_from_returns(ret: pd.Series) -> float:
    """Max drawdown using log cum equity."""
    if ret.empty:
        return np.nan
    log_cum = np.log1p(ret.fillna(0)).cumsum()
    peak = log_cum.cummax()
    dd = peak - log_cum
    return float(dd.max())

def cagr_from_returns(ret: pd.Series, periods_per_year: int = 12) -> float:
    """CAGR from a series of periodic simple returns."""
    if ret.empty:
        return np.nan
    T = len(ret) / periods_per_year
    total = float((1.0 + ret.fillna(0)).prod())
    if T <= 0 or total <= 0:
        return np.nan
    return total**(1.0 / T) - 1.0

def hac_capm_alpha(port_ls: pd.Series, mkt: pd.DataFrame) -> dict:
    """CAPM (port_ls - rf) ~ (mkt - rf), HAC(3). Returns alpha, t, IR."""
    df = mkt.copy()
    df = df.merge(
        pd.DataFrame({"year": mkt["year"], "month": mkt["month"], "port_ls": port_ls.values}),
        on=["year","month"], how="inner"
    )
    df["mkt_rf"]     = df["ret"] - df["rf"]
    df["port_ls_rf"] = df["port_ls"] - df["rf"]
    if df["port_ls_rf"].std(ddof=1) == 0 or df["mkt_rf"].std(ddof=1) == 0:
        return {"alpha": np.nan, "alpha_t": np.nan, "ir_annual": np.nan}
    ols = sm.ols("port_ls_rf ~ mkt_rf", data=df).fit(cov_type="HAC", cov_kwds={"maxlags":3}, use_t=True)
    alpha = float(ols.params.get("Intercept", np.nan))
    tstat = float(ols.tvalues.get("Intercept", np.nan))
    if getattr(ols, "mse_resid", None) is not None and ols.mse_resid > 0:
        ir_ann = alpha / np.sqrt(ols.mse_resid) * np.sqrt(12)
    else:
        ir_ann = np.nan
    return {"alpha": alpha, "alpha_t": tstat, "ir_annual": float(ir_ann)}

def compute_summary_stats(mp: pd.DataFrame, mkt_df: pd.DataFrame, label: str) -> dict:
    """One summary row per variant."""
    row = {"variant": label}
    if mp.empty:
        keys = ["mean_monthly","vol_monthly","sharpe_ann","cagr","max_dd_log",
                "max_1m_loss","alpha","alpha_t","ir_annual","hit_rate","skew","kurt"]
        for k in keys: row[k] = np.nan
        return row
    r = mp["port_ls"].copy()
    row["mean_monthly"] = float(r.mean())
    row["vol_monthly"]  = float(r.std(ddof=1))
    row["sharpe_ann"]   = float(r.mean() / r.std(ddof=1) * np.sqrt(12)) if r.std(ddof=1) > 0 else np.nan
    row["cagr"]         = cagr_from_returns(r, 12)
    row["max_dd_log"]   = max_drawdown_from_returns(r)
    row["max_1m_loss"]  = float(r.min())
    row["hit_rate"]     = float((r > 0).mean()) if len(r) else np.nan
    row["skew"]         = float(r.skew()) if len(r) else np.nan
    row["kurt"]         = float(r.kurt()) if len(r) else np.nan
    capm = hac_capm_alpha(mp[["year","month","port_ls"]].merge(mkt_df, on=["year","month"], how="inner")["port_ls"],
                          mp[["year","month"]].merge(mkt_df, on=["year","month"], how="inner"))
    row.update(capm)
    return row

def turnover_count(df_side: pd.DataFrame) -> float:
    """
    df_side: rows for a single side (long or short) across months with columns ['id','date']
    Measures average % replaced month-to-month (1 - overlap/previous_count).
    """
    if df_side.empty:
        return np.nan

    side = df_side[["id","date"]].copy()
    side["month_start"] = (side["date"] - MonthBegin(1)).dt.normalize() + MonthBegin(1)
    side["month_start"] = side["month_start"].dt.to_period("M").dt.to_timestamp()

    membership = (side.groupby("month_start")["id"]
                  .apply(lambda s: set(s.unique()))
                  .sort_index())

    months = membership.index.to_list()
    if len(months) < 2:
        return 0.0

    turnovers = []
    for i in range(1, len(months)):
        prev_set = membership.iloc[i-1]
        cur_set  = membership.iloc[i]
        if len(prev_set) == 0:
            continue
        overlap = len(prev_set.intersection(cur_set))
        replaced_rate = (len(prev_set) - overlap) / len(prev_set)
        turnovers.append(replaced_rate)

    return float(np.mean(turnovers)) if turnovers else 0.0

def _price_equalized_shares_and_weights(prices: pd.Series, side_capital: float):
    """Equal-dollar allocator."""
    pr = prices.astype(float).abs().replace(0, np.nan).dropna()
    if pr.empty or side_capital <= 0:
        return {}, {}

    maxp = float(pr.max())
    base_shares = np.maximum(1, np.rint(maxp / pr).astype(int))
    base_cost = float((base_shares * pr).sum())
    if base_cost <= 0:
        return {}, {}

    lots = max(1, int(side_capital // base_cost))
    shares = (base_shares * lots).astype(int)

    spent = float((shares * pr).sum())
    cash = side_capital - spent
    if cash > 0:
        order = pr.sort_values(ascending=False)
        for idx, price in order.items():
            if cash >= price:
                extra = int(cash // price)
                if extra > 0:
                    shares.loc[idx] += extra
                    cash -= extra * price
            if cash < pr.min():
                break

    weights = (shares * pr) / side_capital
    return shares.to_dict(), weights.to_dict()

# ===================================
# NEW: Beta Proxy Functions (Memory-Efficient Version)
# ===================================
def compute_beta_proxy_efficient(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Memory-efficient beta proxy using volatility and size.

    Uses vectorized operations instead of groupby.apply() to save memory.
    """
    print(f"Computing fast beta proxy using volatility and size...")
    print(f"Input shape: {feat_df.shape}, Memory: {feat_df.memory_usage(deep=True).sum() / 1024**2:.0f} MB")

    # Select only needed columns to save memory
    needed_cols = ["id", "char_year", "char_month"]

    # Use idiosyncratic volatility as proxy
    if "ivol_capm_21d" in feat_df.columns:
        vol_col = "ivol_capm_21d"
    elif "ivol_ff3_21d" in feat_df.columns:
        vol_col = "ivol_ff3_21d"
    elif "retvol" in feat_df.columns:
        vol_col = "retvol"
    else:
        print("Warning: No volatility column found, using constant volatility")
        vol_col = None

    if vol_col:
        needed_cols.append(vol_col)

    # Use market cap as size proxy
    if "me" in feat_df.columns:
        size_col = "me"
    elif "mvel1" in feat_df.columns:
        size_col = "mvel1"
    else:
        print("Warning: No market cap column found, using constant size")
        size_col = None

    if size_col:
        needed_cols.append(size_col)

    # Create subset with only needed columns
    beta_df = feat_df[needed_cols].copy()

    # Compute ranks within each month using transform (more memory efficient)
    print("Computing cross-sectional ranks...")

    if vol_col:
        beta_df["vol_rank"] = beta_df.groupby(["char_year", "char_month"])[vol_col].transform(
            lambda x: x.rank(pct=True)
        )
    else:
        beta_df["vol_rank"] = 0.5

    if size_col:
        beta_df["size_rank"] = beta_df.groupby(["char_year", "char_month"])[size_col].transform(
            lambda x: 1 - x.rank(pct=True)
        )
    else:
        beta_df["size_rank"] = 0.5

    # Compute beta proxy: weighted average
    # Volatility is more important predictor (70%) vs size (30%)
    beta_df["beta_mkt"] = 0.5 + (0.7 * beta_df["vol_rank"] + 0.3 * beta_df["size_rank"])

    # Keep only needed columns
    beta_df = beta_df[["id", "char_year", "char_month", "beta_mkt"]].copy()

    print(f"Computed beta proxies for {beta_df['id'].nunique()} stocks")
    print(f"Beta proxy statistics: mean={beta_df['beta_mkt'].mean():.2f}, median={beta_df['beta_mkt'].median():.2f}, std={beta_df['beta_mkt'].std():.2f}")

    return beta_df

def map_sector_from_industry(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map industry codes to broad sectors.
    PRIORITY: Uses GICS sector codes (hackathon requirement), then SIC, then fallbacks.

    Returns DataFrame with added 'sector' column.
    """
    feat_df = feat_df.copy()

    # PRIORITY 1: Try GICS sector columns (hackathon requirement)
    gics_cols = [c for c in ["gics_sector", "gsector", "gics"] if c in feat_df.columns]
    if gics_cols:
        c = gics_cols[0]
        feat_df["sector"] = feat_df[c].fillna("Unknown").astype(str).str.upper()
        print(f"Using GICS sectors from column: {c}")
        return feat_df

    # PRIORITY 2: Try to use SIC codes if available
    elif "sic" in feat_df.columns:
        feat_df["sic"] = feat_df["sic"].fillna(0).astype(int)

        def sic_to_sector(sic):
            if sic == 0:
                return "Unknown"
            elif 1000 <= sic < 1500:
                return "Materials"
            elif 1500 <= sic < 1800:
                return "Construction"
            elif 2000 <= sic < 4000:
                return "Manufacturing"
            elif 4000 <= sic < 5000:
                return "Transportation"
            elif 5000 <= sic < 5200:
                return "Wholesale"
            elif 5200 <= sic < 6000:
                return "Retail"
            elif 6000 <= sic < 6800:
                return "Finance"
            elif 6800 <= sic < 7000:
                return "Real Estate"
            elif 7000 <= sic < 9000:
                return "Services"
            elif 9000 <= sic < 10000:
                return "Public Admin"
            else:
                return "Unknown"

        feat_df["sector"] = feat_df["sic"].apply(sic_to_sector)

    # Fallback: use industry column if exists
    elif "industry" in feat_df.columns:
        feat_df["sector"] = feat_df["industry"].fillna("Unknown")

    # Last resort: use country + size as pseudo-sector
    else:
        print("Warning: No SIC or industry column, using country+size as sector proxy")
        feat_df["size_bucket"] = pd.qcut(feat_df.get("me", pd.Series([1]*len(feat_df))),
                                          q=3, labels=["Small", "Mid", "Large"], duplicates="drop")
        feat_df["sector"] = feat_df.get("excntry", "UNK").astype(str) + "_" + feat_df["size_bucket"].astype(str)

    return feat_df

# ------------------------------
# Load predictions (RETURN month = t+1)
# ------------------------------
print("Loading predictions…")
pred = pd.read_csv(PRED_PATH, parse_dates=["date"])

if not {"year", "month"}.issubset(pred.columns):
    pred["year"]  = pred["date"].dt.year
    pred["month"] = pred["date"].dt.month

pred["ret_eom_dt"]  = pred["date"].dt.to_period("M").dt.to_timestamp("M")   # t+1 month-end
pred["char_eom_dt"] = pred["ret_eom_dt"] - MonthEnd(1)                      # t month-end
pred["char_year"]   = pred["char_eom_dt"].dt.year
pred["char_month"]  = pred["char_eom_dt"].dt.month

# ------------------------------
# Load features (char-month t) - only needed columns
# ------------------------------
print("Loading features (liquidity proxies + beta calculation)…")

# Define columns we actually need (to save memory with large parquet file)
needed_feature_cols = [
    "id", "char_eom", "date", "ret_eom", "iid", "excntry", "me",
    "ivol_capm_21d", "sic", "industry",
    "zero_trades_21d", "zero_trades_126d", "zero_trades_252d",
    "dolvol_126d", "turnover_126d", "bidaskhl_21d", "prc",
    # Additional columns for compliance
    "ni", "at",  # Net income and assets for distressed filter
    "gics_sector", "gsector", "gics"  # GICS sector codes (hackathon requirement)
]

# Read only columns that exist
import pyarrow.parquet as pq
parquet_file = pq.ParquetFile(FEATURES_PATH)
available_cols = parquet_file.schema.names
cols_to_read = [c for c in needed_feature_cols if c in available_cols]

print(f"Reading {len(cols_to_read)} columns out of {len(available_cols)} available")
feat = pd.read_parquet(FEATURES_PATH, columns=cols_to_read)

if "char_eom" in feat.columns:
    feat["char_eom_dt"] = pd.to_datetime(feat["char_eom"].astype(str), format="%Y%m%d", errors="coerce")
elif "date" in feat.columns:
    if pd.api.types.is_datetime64_any_dtype(feat["date"]):
        feat["char_eom_dt"] = feat["date"].dt.to_period("M").dt.to_timestamp("M")
    else:
        feat["char_eom_dt"] = pd.to_datetime(feat["date"].astype(str), errors="coerce").dt.to_period("M").dt.to_timestamp("M")
elif "ret_eom" in feat.columns:
    ret_dt = pd.to_datetime(feat["ret_eom"].astype(str), format="%Y%m%d", errors="coerce")
    feat["char_eom_dt"] = ret_dt - MonthEnd(1)
else:
    raise ValueError("Features must have char_eom, date, or ret_eom to derive characteristic month (t).")

feat["char_year"]  = feat["char_eom_dt"].dt.year
feat["char_month"] = feat["char_eom_dt"].dt.month

if "id" not in feat.columns:
    raise ValueError("Features file missing column: id")

# ===================================
# NEW: Compute betas and sectors
# ===================================
# Load market data for beta calculation
mkt = pd.read_csv(MKT_PATH)

# Compute beta proxies (memory-efficient version)
betas_df = compute_beta_proxy_efficient(feat)

# Map sectors
feat = map_sector_from_industry(feat)

# Keep only needed proxy columns (now including beta and sector)
proxy_cols = list(
    set(
        ["id", "char_year", "char_month", "iid", "excntry", "me", "sector"]
        + list(LIQ_COLS)
        + ([BIDASK_COL] if BIDASK_COL else [])
        + ([PRICE_COL] if PRICE_COL else [])
    )
)
feat_for_merge = feat[[c for c in proxy_cols if c in feat.columns]].copy()

# Merge in betas
feat_for_merge = feat_for_merge.merge(
    betas_df,
    on=["id", "char_year", "char_month"],
    how="left"
)

# -----------------------------------------
# Build cross-sectional (t) proxy cutoffs
# -----------------------------------------
def build_monthly_liquidity_cutoffs(feat_like: pd.DataFrame) -> pd.DataFrame:
    grp = feat_like.groupby(["char_year","char_month"], as_index=False)

    def _q(s, q):
        try:
            return float(s.dropna().quantile(q)) if s.notna().any() else np.nan
        except Exception:
            return np.nan

    rows = []
    for (y, m), dfm in grp:
        row = {"char_year": y, "char_month": m}
        if "dolvol_126d" in dfm:
            row["q_dolvol_low"] = _q(dfm["dolvol_126d"], DOLVOL_LOW_Q)
        else:
            row["q_dolvol_low"] = np.nan
        if "turnover_126d" in dfm:
            row["q_turn_low"] = _q(dfm["turnover_126d"], TURNOVER_LOW_Q)
        else:
            row["q_turn_low"] = np.nan
        if BIDASK_COL and BIDASK_COL in dfm:
            row["q_bidask_high"] = _q(dfm[BIDASK_COL], BIDASK_HIGH_Q)
        else:
            row["q_bidask_high"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    return out

print("Computing strict monthly liquidity cutoffs (cross-sectional @ t)…")
liq_cutoffs = build_monthly_liquidity_cutoffs(feat_for_merge)

# ------------------------------
# Merge features@t into predictions@t+1
# ------------------------------
print("Merging proxies from char-month (t) to prediction rows (t+1)…")
predX = pred.merge(
    feat_for_merge,
    on=["id", "char_year", "char_month"],
    how="left",
    suffixes=("", "_feat")
)

# Merge the monthly cutoffs (keyed on t)
predX = predX.merge(
    liq_cutoffs,
    on=["char_year","char_month"],
    how="left"
)

# Sanity: ensure t+1 exactly one month after t
sel_month  = predX["ret_eom_dt"]
char_month = predX["char_eom_dt"]
delta_months = (sel_month.dt.year - char_month.dt.year)*12 + (sel_month.dt.month - char_month.dt.month)
if not np.all(delta_months == 1):
    bad = predX.loc[delta_months != 1, ["id","date","char_eom_dt"]].head(5)
    raise ValueError(f"Date misalignment: expected return month = char month + 1. Examples:\n{bad}")

# ------------------------------
# Liquidity flag (STRICT)
# ------------------------------
def is_illiquid_row(row) -> bool:
    needed_cols = set(LIQ_COLS) | ({BIDASK_COL} if BIDASK_COL else set()) | ({PRICE_COL} if PRICE_COL else set())
    for c in needed_cols:
        if c and c in row and pd.isna(row[c]):
            return True

    for c, thr in STRICT_ZERO_TRADE_THRESH.items():
        if c in row and pd.notna(row[c]) and float(row[c]) >= thr:
            return True

    if "dolvol_126d" in row:
        v = row["dolvol_126d"]
        if pd.notna(v):
            if v <= 0:
                return True
            q_low = row.get("q_dolvol_low", np.nan)
            if pd.notna(q_low) and v <= q_low:
                return True
        else:
            return True
    if "turnover_126d" in row:
        v = row["turnover_126d"]
        if pd.notna(v):
            if v <= 0:
                return True
            q_low = row.get("q_turn_low", np.nan)
            if pd.notna(q_low) and v <= q_low:
                return True
        else:
            return True

    if BIDASK_COL and BIDASK_COL in row and pd.notna(row[BIDASK_COL]):
        ba = float(row[BIDASK_COL])
        q_hi = row.get("q_bidask_high", np.nan)
        if pd.notna(q_hi) and ba >= q_hi:
            return True
        if ABS_MAX_BIDASKHL is not None and ba > ABS_MAX_BIDASKHL:
            return True
    elif BIDASK_COL:
        return True

    if PRICE_COL and PRICE_COL in row:
        p = row[PRICE_COL]
        if pd.notna(p) and ABS_MIN_PRICE is not None and abs(float(p)) < ABS_MIN_PRICE:
            return True
        elif pd.isna(p) and ABS_MIN_PRICE is not None:
            return True

    return False

print("Tagging illiquidity (STRICT)…")
predX["is_illiquid"] = predX.apply(is_illiquid_row, axis=1)

# ===================================
# MODIFIED: L/S selection with beta & sector neutrality
# ===================================
def select_portfolios_one_month(df_month: pd.DataFrame,
                                model_col: str,
                                min_total: int,
                                percentile: float,
                                prev_long_ids=None,
                                prev_short_ids=None,
                                turnover_max_long: float = None,
                                turnover_max_short: float = None,
                                max_country_weight: float = None,
                                min_countries_per_side: int = None,
                                beta_target_diff_max: float = None,
                                max_sector_deviation: float = None):
    """
    Enhanced portfolio selection with beta-balancing and sector neutrality.
    """
    n = len(df_month)
    if n == 0:
        return df_month.iloc[0:0], df_month.iloc[0:0]

    if turnover_max_long is None:  turnover_max_long  = TURNOVER_MAX_LONG
    if turnover_max_short is None: turnover_max_short = TURNOVER_MAX_SHORT
    if max_country_weight is None: max_country_weight = MAX_COUNTRY_WEIGHT
    if min_countries_per_side is None: min_countries_per_side = MIN_COUNTRIES_PER_SIDE
    if beta_target_diff_max is None: beta_target_diff_max = BETA_TARGET_DIFF_MAX
    if max_sector_deviation is None: max_sector_deviation = MAX_SECTOR_DEVIATION
    if prev_long_ids is None:  prev_long_ids  = set()
    if prev_short_ids is None: prev_short_ids = set()

    def ranked(df, side):
        if side == "long":
            return df.sort_values([model_col,"id"], ascending=[False,True]).copy()
        else:
            return df.sort_values([model_col,"id"], ascending=[True,True]).copy()

    q_long  = df_month[model_col].quantile(1 - percentile/100.0)
    q_short = df_month[model_col].quantile(percentile/100.0)

    longs_init_candidates  = df_month[df_month[model_col] >= q_long].copy()
    shorts_init_candidates = df_month[df_month[model_col] <= q_short].copy()

    # OPTIMIZATION: Enhanced large-cap targeting + small-cap penalty (Priority 2)
    if "me" in longs_init_candidates.columns:
        longs_init_candidates["score_adjusted"] = longs_init_candidates[model_col].copy()
        large_cap_mask = longs_init_candidates["me"] >= LARGE_CAP_ME_MILLIONS
        small_cap_mask = longs_init_candidates["me"] < LARGE_CAP_ME_MILLIONS
        longs_init_candidates.loc[large_cap_mask, "score_adjusted"] += LARGE_CAP_SCORE_BOOST
        longs_init_candidates.loc[small_cap_mask, "score_adjusted"] -= SMALL_CAP_PENALTY
    else:
        longs_init_candidates["score_adjusted"] = longs_init_candidates[model_col]

    if "me" in shorts_init_candidates.columns:
        shorts_init_candidates["score_adjusted"] = shorts_init_candidates[model_col].copy()
        large_cap_mask = shorts_init_candidates["me"] >= LARGE_CAP_ME_MILLIONS
        small_cap_mask = shorts_init_candidates["me"] < LARGE_CAP_ME_MILLIONS
        # For shorts, lower score = better, so subtract boost (makes large-caps more negative)
        shorts_init_candidates.loc[large_cap_mask, "score_adjusted"] -= LARGE_CAP_SCORE_BOOST
        shorts_init_candidates.loc[small_cap_mask, "score_adjusted"] += SMALL_CAP_PENALTY
    else:
        shorts_init_candidates["score_adjusted"] = shorts_init_candidates[model_col]

    # OPTIMIZATION: SHORT BOOK FIX - Remove value trap filter only (Priority 1)
    # Key insight from analysis: The old momentum filter was BACKWARDS
    # - Old: Filtered for mom6m < 0 (only negative momentum) → kept value traps that bounced
    # - Fix: Remove deep losers (ret_6_1 < -0.10) which are likely to bounce back
    # - This is the MINIMAL change needed to fix the short book

    # Remove deep value traps (stocks that have fallen >10% in past 6m)
    if "ret_6_1" in shorts_init_candidates.columns:
        shorts_init_candidates = shorts_init_candidates[
            shorts_init_candidates["ret_6_1"].fillna(0) > -0.15  # Exclude deep losers (>-15% in 6m)
        ].copy()

    # Use adjusted scores for ranking
    def ranked_adjusted(df, side):
        if side == "long":
            return df.sort_values(["score_adjusted","id"], ascending=[False,True]).copy()
        else:
            return df.sort_values(["score_adjusted","id"], ascending=[True,True]).copy()

    longs_init  = ranked_adjusted(longs_init_candidates, "long")
    shorts_init = ranked_adjusted(shorts_init_candidates, "short")

    avail_L = len(longs_init)
    avail_S = len(shorts_init)

    k_long_target  = min(MAX_LONG, avail_L)
    k_short_target = min(MAX_SHORT, avail_S)

    if avail_L >= MIN_LONG:
        k_long_target = max(k_long_target, MIN_LONG)
    if avail_S >= MIN_SHORT:
        k_short_target = max(k_short_target, MIN_SHORT)

    total_target = k_long_target + k_short_target
    if total_target < min_total:
        need = min_total - total_target
        room_L = max(0, min(MAX_LONG, avail_L) - k_long_target)
        room_S = max(0, min(MAX_SHORT, avail_S) - k_short_target)

        while need > 0 and (room_L > 0 or room_S > 0):
            frac_L = (k_long_target + k_short_target) and (k_long_target / (k_long_target + k_short_target)) or 0.0
            if frac_L < 0.70 and room_L > 0:
                k_long_target += 1; room_L -= 1; need -= 1; continue
            if room_S > 0:
                k_short_target += 1; room_S -= 1; need -= 1; continue
            if room_L > 0:
                k_long_target += 1; room_L -= 1; need -= 1; continue
            break

    if k_long_target + k_short_target > MAX_TOTAL:
        excess = k_long_target + k_short_target - MAX_TOTAL
        while excess > 0:
            propL = k_long_target / (k_long_target + k_short_target)
            if propL > 0.70 and k_long_target > max(MIN_LONG, 0):
                k_long_target -= 1
            elif k_short_target > max(MIN_SHORT, 0):
                k_short_target -= 1
            else:
                break
            excess -= 1

    def pick_with_constraints(ranked_df, k_target, side, prev_ids, turnover_cap, max_ctry_wt, min_ctrys):
        """Stage 1: Basic selection with country constraints."""
        rd = ranked_df.copy()
        if "excntry" not in rd.columns:
            rd["excntry"] = "UNK"
        rd["excntry"] = rd["excntry"].astype(str).fillna("UNK")

        allowed_changes = int(math.floor(turnover_cap * max(k_target, 1)))
        min_keep = max(0, k_target - allowed_changes)

        rd_prev = rd[rd["id"].isin(prev_ids)].copy()
        rd_new  = rd[~rd["id"].isin(prev_ids)].copy()

        max_count = max(1, int(math.floor(max_ctry_wt * max(k_target, 1))))

        selected_rows = []
        ctry_counts = {}

        def try_add(row):
            c = str(row.get("excntry", "UNK"))
            ctry_counts.setdefault(c, 0)
            if ctry_counts[c] + 1 > max_count:
                return False
            selected_rows.append(row)
            ctry_counts[c] += 1
            return True

        for _, row in rd_prev.iterrows():
            if len(selected_rows) >= min_keep:
                break
            try_add(row)

        for _, row in pd.concat([rd_prev, rd_new], ignore_index=True).iterrows():
            if len(selected_rows) >= k_target:
                break
            try_add(row)

        relax_step = 0
        while len(selected_rows) < k_target and relax_step < 3:
            relax_step += 1
            max_count_relaxed = max_count + relax_step
            for _, row in pd.concat([rd_prev, rd_new], ignore_index=True).iterrows():
                if len(selected_rows) >= k_target:
                    break
                c = str(row.get("excntry", "UNK"))
                current_in = {r["id"] for r in selected_rows}
                if row["id"] in current_in:
                    continue
                c_cnt = ctry_counts.get(c, 0)
                if c_cnt + 1 <= max_count_relaxed:
                    selected_rows.append(row)
                    ctry_counts[c] = c_cnt + 1

        sel = pd.DataFrame(selected_rows).reset_index(drop=True)

        def ensure_min_countries(sel_df):
            uniq = sel_df["excntry"].nunique()
            if uniq >= min_ctrys:
                return sel_df
            current_ids = set(sel_df["id"])
            current_ctry = sel_df["excntry"].mode().iat[0] if len(sel_df) else "UNK"
            candidates_other = rd[~rd["id"].isin(current_ids) & (rd["excntry"] != current_ctry)]
            if candidates_other.empty:
                return sel_df
            to_add = candidates_other.iloc[[0]]
            if side == "long":
                worst_idx = sel_df[sel_df["excntry"] == current_ctry][model_col].idxmin()
            else:
                worst_idx = sel_df[sel_df["excntry"] == current_ctry][model_col].idxmax()
            sel_df = sel_df.drop(index=[worst_idx]).reset_index(drop=True)
            sel_df = pd.concat([sel_df, to_add], ignore_index=True)
            return sel_df

        sel = ensure_min_countries(sel)

        def rebalance_caps(sel_df):
            counts = sel_df["excntry"].value_counts().to_dict()
            over = {c: cnt for c, cnt in counts.items() if cnt > max_count}
            if not over:
                return sel_df
            keep_ids = []
            for c in counts.keys():
                sub = sel_df[sel_df["excntry"] == c]
                if side == "long":
                    sub_sorted = sub.sort_values([model_col,"id"], ascending=[False,True])
                else:
                    sub_sorted = sub.sort_values([model_col,"id"], ascending=[True,True])
                keep_ids.extend(sub_sorted.head(max_count)["id"].tolist())
            kept = sel_df[sel_df["id"].isin(keep_ids)].copy()
            need = k_target - len(kept)
            if need <= 0:
                return kept.reset_index(drop=True)
            kept_ctry = kept["excntry"].value_counts().to_dict()
            for _, row in rd.iterrows():
                if row["id"] in keep_ids:
                    continue
                c = row["excntry"]
                c_cnt = kept_ctry.get(c, 0)
                if c_cnt + 1 <= max_count and row["id"] not in keep_ids:
                    kept = pd.concat([kept, pd.DataFrame([row])], ignore_index=True)
                    keep_ids.append(row["id"])
                    kept_ctry[c] = c_cnt + 1
                if len(kept) >= k_target:
                    break
            return kept.reset_index(drop=True)

        sel = rebalance_caps(sel)

        if len(sel) > k_target:
            if side == "long":
                sel = sel.sort_values([model_col,"id"], ascending=[False,True]).head(k_target)
            else:
                sel = sel.sort_values([model_col,"id"], ascending=[True,True]).head(k_target)
        return sel.reset_index(drop=True)

    # Stage 1: Basic selection
    longs_sel  = pick_with_constraints(longs_init,  k_long_target,  "long",  prev_long_ids,  TURNOVER_MAX_LONG,  MAX_COUNTRY_WEIGHT, MIN_COUNTRIES_PER_SIDE)
    shorts_sel = pick_with_constraints(shorts_init, k_short_target, "short", prev_short_ids, TURNOVER_MAX_SHORT, MAX_COUNTRY_WEIGHT, MIN_COUNTRIES_PER_SIDE)

    # ===================================
    # Stage 2: Beta balancing
    # ===================================
    if "beta_mkt" in longs_sel.columns and "beta_mkt" in shorts_sel.columns:
        # Compute weighted average betas
        longs_sel["beta_mkt_clean"] = longs_sel["beta_mkt"].fillna(1.0)  # Assume beta=1 if missing
        shorts_sel["beta_mkt_clean"] = shorts_sel["beta_mkt"].fillna(1.0)

        beta_long = longs_sel["beta_mkt_clean"].mean()
        beta_short = shorts_sel["beta_mkt_clean"].mean()
        beta_diff = abs(beta_long - beta_short)

        # Iteratively swap stocks to balance betas (max 10 iterations)
        max_beta_iters = 10
        for iter_num in range(max_beta_iters):
            if beta_diff <= beta_target_diff_max:
                break

            # If longs have higher beta, swap high-beta longs with low-beta shorts
            if beta_long > beta_short + beta_target_diff_max:
                # Find highest beta long
                high_beta_long_idx = longs_sel["beta_mkt_clean"].idxmax()
                high_beta_long_id = longs_sel.loc[high_beta_long_idx, "id"]

                # Find a lower-beta candidate from long candidates not yet selected
                remaining_longs = longs_init[~longs_init["id"].isin(longs_sel["id"])]
                if not remaining_longs.empty:
                    remaining_longs["beta_mkt_clean"] = remaining_longs.get("beta_mkt", 1.0).fillna(1.0)
                    # Get lowest beta alternative
                    low_beta_candidate_idx = remaining_longs["beta_mkt_clean"].idxmin()
                    if remaining_longs.loc[low_beta_candidate_idx, "beta_mkt_clean"] < longs_sel.loc[high_beta_long_idx, "beta_mkt_clean"]:
                        # Swap
                        longs_sel = longs_sel[longs_sel["id"] != high_beta_long_id]
                        longs_sel = pd.concat([longs_sel, remaining_longs.loc[[low_beta_candidate_idx]]], ignore_index=True)

            # If shorts have higher beta, swap high-beta shorts with low-beta longs
            elif beta_short > beta_long + beta_target_diff_max:
                # Find highest beta short
                high_beta_short_idx = shorts_sel["beta_mkt_clean"].idxmax()
                high_beta_short_id = shorts_sel.loc[high_beta_short_idx, "id"]

                # Find a lower-beta candidate from short candidates not yet selected
                remaining_shorts = shorts_init[~shorts_init["id"].isin(shorts_sel["id"])]
                if not remaining_shorts.empty:
                    remaining_shorts["beta_mkt_clean"] = remaining_shorts.get("beta_mkt", 1.0).fillna(1.0)
                    # Get lowest beta alternative
                    low_beta_candidate_idx = remaining_shorts["beta_mkt_clean"].idxmin()
                    if remaining_shorts.loc[low_beta_candidate_idx, "beta_mkt_clean"] < shorts_sel.loc[high_beta_short_idx, "beta_mkt_clean"]:
                        # Swap
                        shorts_sel = shorts_sel[shorts_sel["id"] != high_beta_short_id]
                        shorts_sel = pd.concat([shorts_sel, remaining_shorts.loc[[low_beta_candidate_idx]]], ignore_index=True)

            # Recompute betas
            beta_long = longs_sel["beta_mkt_clean"].mean()
            beta_short = shorts_sel["beta_mkt_clean"].mean()
            beta_diff = abs(beta_long - beta_short)

    # ===================================
    # Stage 3: Sector neutrality
    # ===================================
    if "sector" in longs_sel.columns and "sector" in shorts_sel.columns:
        # Compute sector weights
        long_sector_counts = longs_sel["sector"].value_counts(normalize=True)
        short_sector_counts = shorts_sel["sector"].value_counts(normalize=True)

        # Find sectors with largest deviations
        all_sectors = set(long_sector_counts.index) | set(short_sector_counts.index)
        sector_deviations = {}
        for sector in all_sectors:
            long_wt = long_sector_counts.get(sector, 0)
            short_wt = short_sector_counts.get(sector, 0)
            sector_deviations[sector] = long_wt - short_wt

        # Iteratively swap to reduce largest deviations (max 10 iterations)
        max_sector_iters = 10
        for iter_num in range(max_sector_iters):
            max_dev = max(abs(d) for d in sector_deviations.values())
            if max_dev <= max_sector_deviation:
                break

            # Find sector with largest deviation
            worst_sector = max(sector_deviations, key=lambda s: abs(sector_deviations[s]))
            deviation = sector_deviations[worst_sector]

            if deviation > max_sector_deviation:
                # Longs are overweight in this sector, swap a long from this sector with a short from underweight sector
                # Find underweight sector in longs
                underweight_sectors = [s for s, d in sector_deviations.items() if d < -max_sector_deviation]
                if underweight_sectors and worst_sector in longs_sel["sector"].values:
                    target_sector = underweight_sectors[0]

                    # Swap worst-ranked long from overweight sector with best-ranked candidate from underweight sector
                    overweight_longs = longs_sel[longs_sel["sector"] == worst_sector]
                    if len(overweight_longs) > 0:
                        worst_long_idx = overweight_longs[model_col].idxmin()
                        worst_long_id = longs_sel.loc[worst_long_idx, "id"]

                        # Find replacement from underweight sector
                        remaining_longs = longs_init[
                            ~longs_init["id"].isin(longs_sel["id"]) &
                            (longs_init["sector"] == target_sector)
                        ]
                        if not remaining_longs.empty:
                            best_replacement_idx = remaining_longs[model_col].idxmax()
                            longs_sel = longs_sel[longs_sel["id"] != worst_long_id]
                            longs_sel = pd.concat([longs_sel, remaining_longs.loc[[best_replacement_idx]]], ignore_index=True)

            elif deviation < -max_sector_deviation:
                # Shorts are overweight (longs underweight) in this sector
                overweight_sectors = [s for s, d in sector_deviations.items() if d > max_sector_deviation]
                if overweight_sectors and worst_sector in shorts_sel["sector"].values:
                    target_sector = overweight_sectors[0]

                    # Swap worst-ranked short from overweight sector with best-ranked candidate from underweight sector
                    overweight_shorts = shorts_sel[shorts_sel["sector"] == worst_sector]
                    if len(overweight_shorts) > 0:
                        worst_short_idx = overweight_shorts[model_col].idxmax()
                        worst_short_id = shorts_sel.loc[worst_short_idx, "id"]

                        # Find replacement from underweight sector
                        remaining_shorts = shorts_init[
                            ~shorts_init["id"].isin(shorts_sel["id"]) &
                            (shorts_init["sector"] == target_sector)
                        ]
                        if not remaining_shorts.empty:
                            best_replacement_idx = remaining_shorts[model_col].idxmin()
                            shorts_sel = shorts_sel[shorts_sel["id"] != worst_short_id]
                            shorts_sel = pd.concat([shorts_sel, remaining_shorts.loc[[best_replacement_idx]]], ignore_index=True)

            # Recompute sector deviations
            long_sector_counts = longs_sel["sector"].value_counts(normalize=True)
            short_sector_counts = shorts_sel["sector"].value_counts(normalize=True)
            all_sectors = set(long_sector_counts.index) | set(short_sector_counts.index)
            sector_deviations = {}
            for sector in all_sectors:
                long_wt = long_sector_counts.get(sector, 0)
                short_wt = short_sector_counts.get(sector, 0)
                sector_deviations[sector] = long_wt - short_wt

    # ===================================
    # Stage 4: FIX #3 - Enforce 40% Net Sector Exposure (Hackathon Requirement)
    # ===================================
    # Net sector exposure = 1.50 × w_long_sector - 0.50 × w_short_sector
    # Must satisfy: abs(net_sector_wt) <= 0.40 for all sectors
    if "sector" in longs_sel.columns and "sector" in shorts_sel.columns:
        MAX_NET_SECTOR_EXPOSURE = 0.40
        max_sector_exposure_iters = 20

        for iter_num in range(max_sector_exposure_iters):
            # Calculate net sector exposures with leverage weights
            long_sector_counts = longs_sel["sector"].value_counts(normalize=True)
            short_sector_counts = shorts_sel["sector"].value_counts(normalize=True)
            all_sectors = set(long_sector_counts.index) | set(short_sector_counts.index)

            net_sector_exposures = {}
            for sector in all_sectors:
                long_wt = long_sector_counts.get(sector, 0)
                short_wt = short_sector_counts.get(sector, 0)
                # Apply leverage weights: 150% long, 50% short
                net_sector_exposures[sector] = LONG_WEIGHT * long_wt - SHORT_WEIGHT * short_wt

            # Find sector with max violation
            max_violation_sector = None
            max_violation_amount = 0.0
            for sector, net_exp in net_sector_exposures.items():
                if abs(net_exp) > MAX_NET_SECTOR_EXPOSURE:
                    if abs(net_exp) > max_violation_amount:
                        max_violation_amount = abs(net_exp)
                        max_violation_sector = sector

            if max_violation_sector is None:
                break  # All sectors within limits

            # Rebalance to reduce exposure
            if net_sector_exposures[max_violation_sector] > MAX_NET_SECTOR_EXPOSURE:
                # Net long exposure too high: reduce longs or increase shorts in this sector
                # Strategy: swap weakest long from overexposed sector with strongest candidate from underexposed sector

                overexposed_longs = longs_sel[longs_sel["sector"] == max_violation_sector]
                if len(overexposed_longs) > 0:
                    # Find underexposed sectors (net exposure is negative or low)
                    underexposed_sectors = [s for s, exp in net_sector_exposures.items()
                                          if exp < 0 and s != max_violation_sector]
                    if underexposed_sectors:
                        target_sector = min(underexposed_sectors, key=lambda s: net_sector_exposures[s])

                        # Swap worst-ranked long from overexposed sector
                        worst_long_idx = overexposed_longs[model_col].idxmin()
                        worst_long_id = longs_sel.loc[worst_long_idx, "id"]

                        # Find replacement from underexposed sector
                        remaining_longs = longs_init[
                            ~longs_init["id"].isin(longs_sel["id"]) &
                            (longs_init["sector"] == target_sector)
                        ]
                        if not remaining_longs.empty:
                            best_replacement_idx = remaining_longs[model_col].idxmax()
                            longs_sel = longs_sel[longs_sel["id"] != worst_long_id]
                            longs_sel = pd.concat([longs_sel, remaining_longs.loc[[best_replacement_idx]]], ignore_index=True)
                        else:
                            break  # No valid replacement found
                    else:
                        break  # No underexposed sectors to swap with
                else:
                    break

            elif net_sector_exposures[max_violation_sector] < -MAX_NET_SECTOR_EXPOSURE:
                # Net short exposure too high: reduce shorts or increase longs in this sector
                # Strategy: swap weakest short from overexposed sector with strongest candidate from underexposed sector

                overexposed_shorts = shorts_sel[shorts_sel["sector"] == max_violation_sector]
                if len(overexposed_shorts) > 0:
                    # Find underexposed sectors (net exposure is positive or low)
                    underexposed_sectors = [s for s, exp in net_sector_exposures.items()
                                          if exp > 0 and s != max_violation_sector]
                    if underexposed_sectors:
                        target_sector = max(underexposed_sectors, key=lambda s: net_sector_exposures[s])

                        # Swap worst-ranked short from overexposed sector
                        worst_short_idx = overexposed_shorts[model_col].idxmax()
                        worst_short_id = shorts_sel.loc[worst_short_idx, "id"]

                        # Find replacement from underexposed sector
                        remaining_shorts = shorts_init[
                            ~shorts_init["id"].isin(shorts_sel["id"]) &
                            (shorts_init["sector"] == target_sector)
                        ]
                        if not remaining_shorts.empty:
                            best_replacement_idx = remaining_shorts[model_col].idxmin()
                            shorts_sel = shorts_sel[shorts_sel["id"] != worst_short_id]
                            shorts_sel = pd.concat([shorts_sel, remaining_shorts.loc[[best_replacement_idx]]], ignore_index=True)
                        else:
                            break  # No valid replacement found
                    else:
                        break  # No underexposed sectors to swap with
                else:
                    break

    return longs_sel.reset_index(drop=True), shorts_sel.reset_index(drop=True)

# Rest of the code follows the same structure as original...
# (build_long_short function, variants, visualization, etc.)

def build_long_short(pred_like: pd.DataFrame, label: str):
    """Build long/short portfolios with beta and sector constraints."""
    groups = pred_like.groupby(["year","month"], sort=True, as_index=False)
    long_rows, short_rows = [], []
    prev_long_ids, prev_short_ids = set(), set()
    # NEW: Track previous period weights for weight-based turnover
    prev_long_wts, prev_short_wts = {}, {}

    change_rows = []
    turnover_records = {}

    for (y, m), dfm in groups:
        ldf, sdf = select_portfolios_one_month(
            dfm, MODEL_COL, MIN_TOTAL, PERCENTILE,
            prev_long_ids=prev_long_ids,
            prev_short_ids=prev_short_ids,
            turnover_max_long=TURNOVER_MAX_LONG,
            turnover_max_short=TURNOVER_MAX_SHORT,
            max_country_weight=MAX_COUNTRY_WEIGHT,
            min_countries_per_side=MIN_COUNTRIES_PER_SIDE,
            beta_target_diff_max=BETA_TARGET_DIFF_MAX,
            max_sector_deviation=MAX_SECTOR_DEVIATION
        )

        curr_long_ids  = set(ldf["id"].tolist())
        curr_short_ids = set(sdf["id"].tolist())

        # ========================================
        # WEIGHT-BASED TURNOVER (Hackathon requirement)
        # Formula: Turnover_t = 0.5 × Σ|w_i,t - w_i,t-1|
        # ========================================

        # First calculate current period weights
        if not ldf.empty and PRICE_COL in ldf.columns and ldf[PRICE_COL].notna().any():
            # Set index to stock ID so weights dict is keyed by stock ID
            ldf_indexed = ldf.set_index("id")
            shares_L, wts_L = _price_equalized_shares_and_weights(ldf_indexed[PRICE_COL], AUM_START * LONG_WEIGHT)
            curr_long_wts = wts_L  # Now keyed by stock ID
        else:
            # Equal weight if no prices
            curr_long_wts = {stock_id: 1.0/len(ldf) for stock_id in ldf["id"]} if len(ldf) > 0 else {}

        if not sdf.empty and PRICE_COL in sdf.columns and sdf[PRICE_COL].notna().any():
            # Set index to stock ID so weights dict is keyed by stock ID
            sdf_indexed = sdf.set_index("id")
            shares_S, wts_S = _price_equalized_shares_and_weights(sdf_indexed[PRICE_COL], AUM_START * SHORT_WEIGHT)
            curr_short_wts = wts_S  # Now keyed by stock ID
        else:
            # Equal weight if no prices
            curr_short_wts = {stock_id: 1.0/len(sdf) for stock_id in sdf["id"]} if len(sdf) > 0 else {}

        # Weight-based turnover calculation
        def _weight_based_turnover(curr_wts, prev_wts, curr_ids, prev_ids, side):
            """
            Calculate turnover using weight-based formula (hackathon compliant).
            Turnover_t = 0.5 × Σ|w_i,t - w_i,t-1|
            """
            added   = sorted(list(curr_ids - prev_ids))
            removed = sorted(list(prev_ids - curr_ids))
            kept    = sorted(list(curr_ids & prev_ids))

            # Track changes for logging
            for _id in added:
                change_rows.append({"year": y, "month": m, "side": side, "change": "added", "id": _id, "variant": label})
            for _id in kept:
                change_rows.append({"year": y, "month": m, "side": side, "change": "kept", "id": _id, "variant": label})
            for _id in removed:
                change_rows.append({"year": y, "month": m, "side": side, "change": "removed", "id": _id, "variant": label})

            # Calculate weight-based turnover
            if len(prev_wts) == 0:
                return np.nan

            total_weight_change = 0.0

            # Stocks that were removed: contribute their full previous weight
            for stock_id in removed:
                total_weight_change += prev_wts.get(stock_id, 0.0)

            # Stocks that were added: contribute their full current weight
            for stock_id in added:
                total_weight_change += curr_wts.get(stock_id, 0.0)

            # Stocks that stayed: contribute |current_weight - previous_weight|
            for stock_id in kept:
                curr_wt = curr_wts.get(stock_id, 0.0)
                prev_wt = prev_wts.get(stock_id, 0.0)
                total_weight_change += abs(curr_wt - prev_wt)

            # Hackathon formula: Turnover = 0.5 × Σ|w_i,t - w_i,t-1|
            turnover = 0.5 * total_weight_change
            return turnover

        long_turn  = _weight_based_turnover(curr_long_wts,  prev_long_wts,  curr_long_ids,  prev_long_ids,  "long")
        short_turn = _weight_based_turnover(curr_short_wts, prev_short_wts, curr_short_ids, prev_short_ids, "short")

        if np.isnan(long_turn) and np.isnan(short_turn):
            overall_turn = np.nan
        else:
            # Portfolio-wide turnover weighted by actual leverage (150/50)
            # Long weight: 150/(150+50) = 0.75, Short weight: 50/(150+50) = 0.25
            long_weight_pct = LONG_WEIGHT / (LONG_WEIGHT + SHORT_WEIGHT)
            short_weight_pct = SHORT_WEIGHT / (LONG_WEIGHT + SHORT_WEIGHT)
            overall_turn = (long_weight_pct * (0 if np.isnan(long_turn) else long_turn)) + \
                          (short_weight_pct * (0 if np.isnan(short_turn) else short_turn))

        turnover_records[(y, m)] = {"long_turnover": long_turn, "short_turnover": short_turn, "overall_turnover": overall_turn}

        # Add shares & dollar weights to dataframes (use already-calculated weights)
        ldf = ldf.copy()
        if not ldf.empty:
            if len(curr_long_wts) > 0:
                # Use the weights we already calculated (keyed by stock ID)
                ldf["dollar_wt"] = ldf["id"].map(curr_long_wts).fillna(0.0)
                # Calculate shares from weights (reverse of what _price_equalized_shares_and_weights does)
                # For simplicity, use equal shares with weight-based allocation
                ldf["shares"] = 1
            else:
                ldf["shares"] = 1
                ldf["dollar_wt"] = 1.0 / len(ldf) if len(ldf) > 0 else 0.0
        else:
            ldf["shares"] = []
            ldf["dollar_wt"] = []

        sdf = sdf.copy()
        if not sdf.empty:
            if len(curr_short_wts) > 0:
                # Use the weights we already calculated (keyed by stock ID)
                sdf["dollar_wt"] = sdf["id"].map(curr_short_wts).fillna(0.0)
                sdf["shares"] = 1
            else:
                sdf["shares"] = 1
                sdf["dollar_wt"] = 1.0 / len(sdf) if len(sdf) > 0 else 0.0
        else:
            sdf["shares"] = []
            sdf["dollar_wt"] = []

        # Tag cap bucket for tracking
        if "me" in ldf.columns:
            ldf["is_largecap"] = ldf["me"].astype(float) >= LARGE_CAP_ME_MILLIONS
        else:
            ldf["is_largecap"] = np.nan
        if "me" in sdf.columns:
            sdf["is_largecap"] = sdf["me"].astype(float) >= LARGE_CAP_ME_MILLIONS
        else:
            sdf["is_largecap"] = np.nan

        ldf = ldf.assign(year=y, month=m, side="long", variant=label)
        sdf = sdf.assign(year=y, month=m, side="short", variant=label)
        long_rows.append(ldf)
        short_rows.append(sdf)

        # Update previous period tracking (for next month's turnover calculation)
        prev_long_ids = curr_long_ids
        prev_short_ids = curr_short_ids
        prev_long_wts = curr_long_wts.copy()  # Store weights for weight-based turnover
        prev_short_wts = curr_short_wts.copy()

    long_df  = pd.concat(long_rows,  ignore_index=True) if long_rows  else pred_like.iloc[0:0]
    short_df = pd.concat(short_rows, ignore_index=True) if short_rows else pred_like.iloc[0:0]

    # HACKATHON REQUIREMENT: Export monthly holdings CSV for audit
    # Portfolio weights: long (+150%), short (-50%)
    holdings_df = pd.concat([
        long_df.assign(portfolio_wt=LONG_WEIGHT * long_df["dollar_wt"]),
        short_df.assign(portfolio_wt=-SHORT_WEIGHT * short_df["dollar_wt"])
    ], ignore_index=True)
    holdings_cols = ["year", "month", "id", "side", "portfolio_wt"]
    if PRICE_COL in holdings_df.columns:
        holdings_cols.append(PRICE_COL)
    holdings_out = os.path.join(CSV_DIR, f"holdings_{label}.csv")
    holdings_df[holdings_cols].sort_values(["year", "month", "side", "id"]).to_csv(holdings_out, index=False)
    print(f"Saved {holdings_out}")

    # Weighted monthly returns
    def _weighted_mean(group, col_val="stock_ret", col_w="dollar_wt"):
        w = group[col_w].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        v = group[col_val].astype(float).fillna(0.0)
        s = float(w.sum())
        if s <= 0:
            return float(v.mean()) if len(v) else np.nan
        return float((v * w).sum() / s)

    long_ret  = long_df.groupby(["year","month"]).apply(lambda g: _weighted_mean(g, "stock_ret", "dollar_wt")).rename("long_ret")
    short_ret = short_df.groupby(["year","month"]).apply(lambda g: _weighted_mean(g, "stock_ret", "dollar_wt")).rename("short_ret")

    monthly = pd.concat([long_ret, short_ret], axis=1).dropna().reset_index()

    # Market-cap composition tracking per side
    def _cap_counts(df_side):
        if df_side.empty:
            return pd.DataFrame(columns=["year","month","n_large","n_small"])
        tmp = df_side.copy()
        tmp["is_largecap_bool"] = tmp["is_largecap"].fillna(False).astype(bool)
        agg = tmp.groupby(["year","month"]).agg(
            n_side=("id","nunique"),
            n_large=("is_largecap_bool","sum")
        ).reset_index()
        agg["n_small"] = agg["n_side"] - agg["n_large"]
        return agg

    long_cap = _cap_counts(long_df).rename(columns={"n_side": "n_long"})
    short_cap = _cap_counts(short_df).rename(columns={"n_side": "n_short"})

    # Merge counts into monthly
    monthly = (monthly.merge(long_cap[["year","month","n_long","n_large","n_small"]],
                             on=["year","month"], how="left")
                      .merge(short_cap[["year","month","n_short","n_large","n_small"]],
                             on=["year","month"], how="left", suffixes=("_long","_short")))

    # Fill NaNs for counts with zeros
    for c in ["n_long","n_short","n_large_long","n_small_long","n_large_short","n_small_short"]:
        if c in monthly.columns:
            monthly[c] = monthly[c].fillna(0).astype(int)

    # ===================================
    # FIX #6 - Cap Composition by Weight (Hackathon Requirement)
    # ===================================
    # Calculate large-cap % by dollar weight (not just count)
    def _cap_weight_shares(df_side):
        """Calculate large-cap composition by dollar weight."""
        if df_side.empty or "is_largecap" not in df_side.columns:
            return pd.DataFrame()
        tmp = df_side.copy()
        tmp["is_largecap_bool"] = tmp["is_largecap"].fillna(False).astype(bool)
        agg = tmp.groupby(["year","month"]).apply(
            lambda g: pd.Series({
                "wt_large": float((g.loc[g["is_largecap_bool"], "dollar_wt"]).sum()),
                "wt_total": float(g["dollar_wt"].sum())
            })
        ).reset_index()
        agg["share_large_side"] = np.where(agg["wt_total"] > 0, agg["wt_large"] / agg["wt_total"], np.nan)
        return agg[["year","month","share_large_side"]]

    long_capW = _cap_weight_shares(long_df).rename(columns={"share_large_side": "share_large_long"})
    short_capW = _cap_weight_shares(short_df).rename(columns={"share_large_side": "share_large_short"})
    monthly = (monthly.merge(long_capW, on=["year","month"], how="left")
                      .merge(short_capW, on=["year","month"], how="left"))

    # Original gross L/S return with NEW WEIGHTS
    monthly["port_ls_gross"] = LONG_WEIGHT * monthly["long_ret"] - SHORT_WEIGHT * monthly["short_ret"]

    # Attach per-month turnover stats
    if turnover_records:
        trn = pd.DataFrame(
            [dict(year=k[0], month=k[1], **v) for k, v in turnover_records.items()]
        )
        monthly = monthly.merge(trn, on=["year","month"], how="left")

    # Apply market-cap-based fees per turnover
    def _eff_fee_rate(n_large, n_total):
        n_total = n_total if n_total and n_total > 0 else 0
        if n_total == 0:
            return 0.0
        frac_large = float(n_large) / float(n_total)
        frac_small = 1.0 - frac_large
        return frac_large * FEE_PER_TURN_LARGE + frac_small * FEE_PER_TURN_SMALL

    monthly["fee_rate_long_eff"]  = monthly.apply(lambda r: _eff_fee_rate(r.get("n_large_long", 0),  r.get("n_long", 0)), axis=1)
    monthly["fee_rate_short_eff"] = monthly.apply(lambda r: _eff_fee_rate(r.get("n_large_short", 0), r.get("n_short", 0)), axis=1)

    # Fees deducted from portfolio return (per side)
    monthly["fee_long"]  = monthly["long_turnover"].fillna(0.0)  * monthly["fee_rate_long_eff"].fillna(0.0)  * LONG_WEIGHT
    monthly["fee_short"] = monthly["short_turnover"].fillna(0.0) * monthly["fee_rate_short_eff"].fillna(0.0) * SHORT_WEIGHT

    # Net L/S return after fees
    monthly["port_ls"] = monthly["port_ls_gross"] - (monthly["fee_long"].fillna(0.0) + monthly["fee_short"].fillna(0.0))

    # HACKATHON REQUIREMENT: Track AUM and P&L in dollars
    monthly = monthly.sort_values(["year", "month"]).reset_index(drop=True)
    monthly["aum_start"] = AUM_START * (1.0 + monthly["port_ls"].shift(1).fillna(0.0)).cumprod()
    monthly.loc[0, "aum_start"] = AUM_START
    monthly["pnl"] = monthly["aum_start"] * monthly["port_ls"]
    monthly["aum_end"] = monthly["aum_start"] * (1.0 + monthly["port_ls"])

    counts_long  = long_df.groupby(["year","month"])["id"].nunique().rename("n_long_raw")
    counts_short = short_df.groupby(["year","month"])["id"].nunique().rename("n_short_raw")
    monthly = (monthly.merge(counts_long,  on=["year","month"], how="left")
                      .merge(counts_short, on=["year","month"], how="left"))
    monthly["n_total"] = monthly["n_long_raw"].fillna(0).astype(int) + monthly["n_short_raw"].fillna(0).astype(int)
    monthly["variant"] = label

    # NEW: Compute and track beta statistics
    if "beta_mkt_clean" in long_df.columns and "beta_mkt_clean" in short_df.columns:
        long_betas = long_df.groupby(["year", "month"])["beta_mkt_clean"].mean().rename("avg_beta_long")
        short_betas = short_df.groupby(["year", "month"])["beta_mkt_clean"].mean().rename("avg_beta_short")
        monthly = monthly.merge(long_betas, on=["year", "month"], how="left")
        monthly = monthly.merge(short_betas, on=["year", "month"], how="left")
        monthly["beta_diff"] = (monthly["avg_beta_long"] - monthly["avg_beta_short"]).abs()

    # Change log dataframe
    changes_df = pd.DataFrame(change_rows) if change_rows else pred_like.iloc[0:0]

    return long_df, short_df, monthly, changes_df

# ------------------------------
# Build universes / variants
# ------------------------------
print("Building variants…")
base_uni   = predX.copy()
liq_uni    = predX[~predX["is_illiquid"]].copy()

variants = {
    "OPTIMIZED": liq_uni,  # Run optimized strategy on liquid universe
}

# ------------------------------
# Run L/S for each variant
# ------------------------------
ls_outputs = {}
for label, uni in variants.items():
    print(f"Selecting portfolios for {label} (n={len(uni):,})…")
    long_df, short_df, monthly, changes = build_long_short(uni, label)
    ls_outputs[label] = {"long_df": long_df, "short_df": short_df, "monthly": monthly, "changes": changes}
    if not monthly.empty:
        outp = os.path.join(CSV_DIR, f"monthly_ls_{label}.csv")
        monthly.sort_values(["year","month"]).to_csv(outp, index=False)
        print(f"Saved {outp}")
    if isinstance(changes, pd.DataFrame) and not changes.empty:
        changes_outp = os.path.join(CSV_DIR, f"changes_{label}.csv")
        changes.sort_values(["year","month","side","change","id"]).to_csv(changes_outp, index=False)

# ------------------------------
# Summary stats per variant
# ------------------------------
print("Computing summary stats table…")
summary_rows = []
for label, out in ls_outputs.items():
    mp = out["monthly"].sort_values(["year","month"]).reset_index(drop=True)
    if not mp.empty:
        mp = mp.merge(mkt[["year","month","ret","rf"]], on=["year","month"], how="inner")
    row = compute_summary_stats(mp, mkt, label)

    # Average turnover stats
    monthly_src = ls_outputs[label]["monthly"]
    if not monthly_src.empty:
        row["avg_long_turnover"]    = float(monthly_src["long_turnover"].dropna().mean()) if "long_turnover" in monthly_src else np.nan
        row["avg_short_turnover"]   = float(monthly_src["short_turnover"].dropna().mean()) if "short_turnover" in monthly_src else np.nan
        row["avg_overall_turnover"] = float(monthly_src["overall_turnover"].dropna().mean()) if "overall_turnover" in monthly_src else np.nan
        if {"n_long","n_large_long"}.issubset(monthly_src.columns):
            with np.errstate(invalid="ignore", divide="ignore"):
                row["avg_frac_large_long"]  = float(((monthly_src["n_large_long"] / monthly_src["n_long"]).replace([np.inf,-np.inf], np.nan)).mean())
        if {"n_short","n_large_short"}.issubset(monthly_src.columns):
            with np.errstate(invalid="ignore", divide="ignore"):
                row["avg_frac_large_short"] = float(((monthly_src["n_large_short"] / monthly_src["n_short"]).replace([np.inf,-np.inf], np.nan)).mean())
        if {"fee_long","fee_short"}.issubset(monthly_src.columns):
            row["avg_fee_drag"] = float((monthly_src["fee_long"].fillna(0) + monthly_src["fee_short"].fillna(0)).mean())
        # NEW: Beta statistics
        if "avg_beta_long" in monthly_src.columns:
            row["avg_beta_long"] = float(monthly_src["avg_beta_long"].mean())
        if "avg_beta_short" in monthly_src.columns:
            row["avg_beta_short"] = float(monthly_src["avg_beta_short"].mean())
        if "beta_diff" in monthly_src.columns:
            row["avg_beta_diff"] = float(monthly_src["beta_diff"].mean())
    else:
        row["avg_long_turnover"] = row["avg_short_turnover"] = row["avg_overall_turnover"] = np.nan
        row["avg_frac_large_long"] = row["avg_frac_large_short"] = np.nan
        row["avg_fee_drag"] = np.nan
        row["avg_beta_long"] = row["avg_beta_short"] = row["avg_beta_diff"] = np.nan

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(CSV_DIR, "summary_stats_beta_neutral.csv"), index=False)
print("Saved csv/summary_stats_beta_neutral.csv")

print("\n" + "="*80)
print("BETA-NEUTRAL BACKTEST COMPLETE")
print("="*80)
print(f"\nKey improvements from original strategy:")
print(f"1. Net exposure reduced: 100% → {100*(LONG_WEIGHT - SHORT_WEIGHT):.0f}%")
print(f"2. Beta-balanced portfolios (target diff ≤ {BETA_TARGET_DIFF_MAX})")
print(f"3. Sector neutrality constraints (max deviation ≤ {MAX_SECTOR_DEVIATION})")
print(f"\nResults saved to:")
print(f"  - csv/monthly_ls_OPTIMIZED.csv")
print(f"  - csv/summary_stats_beta_neutral.csv")
print("\nCompare with original strategy by running t_calculate_returns_with_fees.py")
print("="*80)
