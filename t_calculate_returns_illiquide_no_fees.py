# backtest.py
# -*- coding: utf-8 -*-
"""
FIAM backtest & diagnostics (L/S everywhere) with CSV outputs for all visuals
- Builds L/S for variants: BASE, LIQ_ONLY, ILLIQ_ONLY
- Adds liquidity pipeline flags (time-t, no look-ahead) with STRICT proxy rules
- Extracts metrics & writes a single-row summary per variant (Sharpe, CAPM alpha, t, IR, CAGR, MaxDD, etc.)
- Saves CSVs that back every figure: equity curves, SPX vs ours, rolling stats, liquidity %, pool vs picked (illiquidity), etc.
- Figures go to ./figs ; data tables go to ./csv

NOTE: Any prior "warrant" handling has been removed. Former IID/ID "*W" are NOT treated specially.
"""

import os
import math
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd, MonthBegin  # <-- MonthBegin added for turnover logic

# ------------------------------
# Config
# ------------------------------
PRED_PATH     = "output_final.csv"          # predictions file (must include: date, year, month, id, stock_ret, and MODEL_COL)
FEATURES_PATH = "data/ret_sample.parquet"   # original features universe (for liquidity proxies, country, iid, etc.)
MKT_PATH      = "data/mkt_ind.csv"          # market data (rf, ret) monthly

MODEL_COL     = "blend"               # prediction column to use
PERCENTILE    = 1                     # top/bottom PERCENTILE% as initial threshold

# New explicit name-count limits (per side and total)
MIN_TOTAL     = 100                   # min total names (try to reach if availability allows)
MAX_TOTAL     = 250                   # max total names across long+short
MIN_LONG      = 70                    # min long names (subject to availability)
MIN_SHORT     = 30                    # min short names (subject to availability)
MAX_LONG      = 175                   # max long names
MAX_SHORT     = 75                    # max short names

# ==============================
# STRICT Liquidity rules (time-t)
# ==============================
STRICT_ZERO_TRADE_THRESH = {
    "zero_trades_21d": 0.2,
    "zero_trades_126d": 0.3,
    "zero_trades_252d": 0.3
}

# Cross-sectional quantiles (computed each char-month t on the *feature* file)
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
BIDASK_COL = "bidaskhl_21d"   # optional
PRICE_COL  = "prc"            # optional

# ---- Concentration + turnover caps ----
MAX_COUNTRY_WEIGHT = 0.75
MIN_COUNTRIES_PER_SIDE = 2
TURNOVER_MAX_LONG  = 0.45
TURNOVER_MAX_SHORT = 0.45

# ---- Constant 70/30 exposure weights for the L/S strategy ----
LONG_WEIGHT  = 1.50
SHORT_WEIGHT = 0.50

# ---- NEW: Starting AUM ----
AUM_START = 500_000_000.0

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

# --- NEW: turnover_count identical to your reference file (membership-change method) ---
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

# --- NEW: price-based equal-dollar allocator ---
def _price_equalized_shares_and_weights(prices: pd.Series, side_capital: float):
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
# Load features (char-month t) — no look-ahead
# ------------------------------
print("Loading features (liquidity proxies)…")
feat = pd.read_parquet(FEATURES_PATH)

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

# Keep only needed proxy columns
proxy_cols = list(
    set(
        ["id", "char_year", "char_month", "iid", "excntry"]
        + list(LIQ_COLS)
        + ([BIDASK_COL] if BIDASK_COL else [])
        + ([PRICE_COL] if PRICE_COL else [])
    )
)
feat_for_merge = feat[[c for c in proxy_cols if c in feat.columns]].copy()

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

# ------------------------------
# L/S selection (per month)
# ------------------------------
def select_portfolios_one_month(df_month: pd.DataFrame,
                                model_col: str,
                                min_total: int,
                                percentile: float,
                                prev_long_ids=None,
                                prev_short_ids=None,
                                turnover_max_long: float = None,
                                turnover_max_short: float = None,
                                max_country_weight: float = None,
                                min_countries_per_side: int = None):
    n = len(df_month)
    if n == 0:
        return df_month.iloc[0:0], df_month.iloc[0:0]

    if turnover_max_long is None:  turnover_max_long  = TURNOVER_MAX_LONG
    if turnover_max_short is None: turnover_max_short = TURNOVER_MAX_SHORT
    if max_country_weight is None: max_country_weight = MAX_COUNTRY_WEIGHT
    if min_countries_per_side is None: min_countries_per_side = MIN_COUNTRIES_PER_SIDE
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

    longs_init  = ranked(longs_init_candidates, "long")
    shorts_init = ranked(shorts_init_candidates, "short")

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

    longs_sel  = pick_with_constraints(longs_init,  k_long_target,  "long",  prev_long_ids,  TURNOVER_MAX_LONG,  MAX_COUNTRY_WEIGHT, MIN_COUNTRIES_PER_SIDE)
    shorts_sel = pick_with_constraints(shorts_init, k_short_target, "short", prev_short_ids, TURNOVER_MAX_SHORT, MAX_COUNTRY_WEIGHT, MIN_COUNTRIES_PER_SIDE)

    return longs_sel.reset_index(drop=True), shorts_sel.reset_index(drop=True)

def build_long_short(pred_like: pd.DataFrame, label: str):
    groups = pred_like.groupby(["year","month"], sort=True, as_index=False)
    long_rows, short_rows = [], []
    prev_long_ids, prev_short_ids = set(), set()

    # change logs and turnover tracking containers
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
            min_countries_per_side=MIN_COUNTRIES_PER_SIDE
        )

        curr_long_ids  = set(ldf["id"].tolist())
        curr_short_ids = set(sdf["id"].tolist())

        # --- Turnover per side using the reference logic (membership change only) ---
        def _side_changes_and_turnover(curr, prev, side):
            added   = sorted(list(curr - prev))
            removed = sorted(list(prev - curr))
            kept    = sorted(list(curr & prev))
            prev_count = len(prev)
            # Reference turnover: (prev_count - overlap) / prev_count = removed / prev_count
            if prev_count > 0:
                overlap = len(kept)
                turnover = (prev_count - overlap) / prev_count
            else:
                turnover = np.nan
            # still log changes (added/kept/removed) for your CSVs
            for _id in added:
                change_rows.append({"year": y, "month": m, "side": side, "change": "added", "id": _id, "variant": label})
            for _id in kept:
                change_rows.append({"year": y, "month": m, "side": side, "change": "kept", "id": _id, "variant": label})
            for _id in removed:
                change_rows.append({"year": y, "month": m, "side": side, "change": "removed", "id": _id, "variant": label})
            return turnover

        long_turn  = _side_changes_and_turnover(curr_long_ids,  prev_long_ids,  "long")
        short_turn = _side_changes_and_turnover(curr_short_ids, prev_short_ids, "short")

        if np.isnan(long_turn) and np.isnan(short_turn):
            overall_turn = np.nan
        else:
            overall_turn = (0.70 * (0 if np.isnan(long_turn) else long_turn)) + (0.30 * (0 if np.isnan(short_turn) else short_turn))

        turnover_records[(y, m)] = {"long_turnover": long_turn, "short_turnover": short_turn, "overall_turnover": overall_turn}

        # Realistic shares & dollar weights
        if not ldf.empty and PRICE_COL in ldf.columns and ldf[PRICE_COL].notna().any():
            shares_L, wts_L = _price_equalized_shares_and_weights(ldf[PRICE_COL], AUM_START * LONG_WEIGHT)
            ldf = ldf.copy()
            ldf["shares"] = ldf["id"].map(shares_L).fillna(0).astype(int)
            ldf["dollar_wt"] = ldf["id"].map(wts_L).fillna(0.0)
        else:
            ldf = ldf.copy()
            if len(ldf) > 0:
                ldf["shares"] = 1
                ldf["dollar_wt"] = 1.0 / len(ldf)
            else:
                ldf["shares"] = []
                ldf["dollar_wt"] = []

        if not sdf.empty and PRICE_COL in sdf.columns and sdf[PRICE_COL].notna().any():
            shares_S, wts_S = _price_equalized_shares_and_weights(sdf[PRICE_COL], AUM_START * SHORT_WEIGHT)
            sdf = sdf.copy()
            sdf["shares"] = sdf["id"].map(shares_S).fillna(0).astype(int)
            sdf["dollar_wt"] = sdf["id"].map(wts_S).fillna(0.0)
        else:
            sdf = sdf.copy()
            if len(sdf) > 0:
                sdf["shares"] = 1
                sdf["dollar_wt"] = 1.0 / len(sdf)
            else:
                sdf["shares"] = []
                sdf["dollar_wt"] = []

        ldf = ldf.assign(year=y, month=m, side="long", variant=label)
        sdf = sdf.assign(year=y, month=m, side="short", variant=label)
        long_rows.append(ldf)
        short_rows.append(sdf)
        prev_long_ids = curr_long_ids
        prev_short_ids = curr_short_ids

    long_df  = pd.concat(long_rows,  ignore_index=True) if long_rows  else pred_like.iloc[0:0]
    short_df = pd.concat(short_rows, ignore_index=True) if short_rows else pred_like.iloc[0:0]

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
    monthly["port_ls"] = LONG_WEIGHT * monthly["long_ret"] - SHORT_WEIGHT * monthly["short_ret"]
    counts_long  = long_df.groupby(["year","month"])["id"].nunique().rename("n_long")
    counts_short = short_df.groupby(["year","month"])["id"].nunique().rename("n_short")
    monthly = (monthly.merge(counts_long,  on=["year","month"], how="left")
                      .merge(counts_short, on=["year","month"], how="left"))
    monthly["n_total"] = monthly["n_long"] + monthly["n_short"]
    monthly["variant"] = label

    # Attach per-month turnover stats (membership-change method)
    if turnover_records:
        trn = pd.DataFrame(
            [dict(year=k[0], month=k[1], **v) for k, v in turnover_records.items()]
        )
        monthly = monthly.merge(trn, on=["year","month"], how="left")

    # Change log dataframe
    changes_df = pd.DataFrame(change_rows) if change_rows else pred_like.iloc[0:0]

    return long_df, short_df, monthly, changes_df

# ------------------------------
# Build universes / variants
# ------------------------------
print("Building variants…")
base_uni   = predX.copy()
liq_uni    = predX[~predX["is_illiquid"]].copy()
illiq_uni  = predX[predX["is_illiquid"]].copy()

variants = {
    "BASE": base_uni,
    "LIQ_ONLY": liq_uni,
    "ILLIQ_ONLY": illiq_uni
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
    if isinstance(changes, pd.DataFrame) and not changes.empty:
        changes_outp = os.path.join(CSV_DIR, f"changes_{label}.csv")
        changes.sort_values(["year","month","side","change","id"]).to_csv(changes_outp, index=False)

# Load market once
mkt = pd.read_csv(MKT_PATH)  # columns: year, month, ret, rf

# ------------------------------
# Summary stats per variant (single-row table)
# ------------------------------
print("Computing summary stats table…")
summary_rows = []
for label, out in ls_outputs.items():
    mp = out["monthly"].sort_values(["year","month"]).reset_index(drop=True)
    if not mp.empty:
        mp = mp.merge(mkt[["year","month","ret","rf"]], on=["year","month"], how="inner")
    row = compute_summary_stats(mp, mkt, label)

    # --- Average turnover stats using the new definition ---
    monthly_src = ls_outputs[label]["monthly"]
    if not monthly_src.empty:
        row["avg_long_turnover"]    = float(monthly_src["long_turnover"].dropna().mean()) if "long_turnover" in monthly_src else np.nan
        row["avg_short_turnover"]   = float(monthly_src["short_turnover"].dropna().mean()) if "short_turnover" in monthly_src else np.nan
        row["avg_overall_turnover"] = float(monthly_src["overall_turnover"].dropna().mean()) if "overall_turnover" in monthly_src else np.nan
    else:
        row["avg_long_turnover"] = row["avg_short_turnover"] = row["avg_overall_turnover"] = np.nan

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(CSV_DIR, "summary_stats_by_variant.csv"), index=False)
print("Saved csv/summary_stats_by_variant.csv")

# ------------------------------
# Visualizations + matching CSVs
# ------------------------------
print("Building L–S visualizations & CSVs…")

# A) L–S Equity curves: LIQ_ONLY vs ILLIQ_ONLY vs BASE
def ls_curve_from_monthly(mp: pd.DataFrame):
    mp = mp.sort_values(["year","month"]).reset_index(drop=True)
    mp["equity_ls"] = (equity_curve(mp["port_ls"]) * AUM_START).shift(1, fill_value=AUM_START)
    return mp

curves_table = []
for key in ["LIQ_ONLY", "ILLIQ_ONLY", "BASE"]:
    mp = ls_outputs[key]["monthly"].copy()
    if mp.empty:
        continue
    mp = ls_curve_from_monthly(mp)
    mp["variant"] = key
    curves_table.append(mp[["year","month","port_ls","equity_ls","variant"]])
if curves_table:
    curves_out = pd.concat(curves_table, ignore_index=True)
    curves_out.to_csv(os.path.join(CSV_DIR, "ls_equity_liquid_vs_illiquid_vs_base.csv"), index=False)

plt.figure(figsize=(12,6))
for key in ["LIQ_ONLY", "ILLIQ_ONLY", "BASE"]:
    mp = ls_outputs[key]["monthly"]
    if len(mp)==0: continue
    ec = (equity_curve(mp.sort_values(["year","month"])["port_ls"]) * AUM_START).shift(1, fill_value=AUM_START)
    plt.plot(range(len(ec)), ec, label=key, linewidth=2)
plt.ylabel(f"Equity (L–S, ${AUM_START:,.0f} start)")
plt.xlabel("Month index (aligned per series)")
plt.title("L–S Equity Curves: Liquid vs Illiquid vs Base")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"ls_equity_liquid_vs_illiquid_vs_base.png"), dpi=PLOTS_DPI, bbox_inches="tight")
plt.close()

# B) S&P500 vs our L–S (BASE / LIQ_ONLY / ILLIQ_ONLY) + CSVs
def align_with_market(mp):
    out = mp.merge(mkt[["year","month","ret","rf"]], on=["year","month"], how="inner")
    out = out.sort_values(["year","month"]).reset_index(drop=True)
    out["mkt_ec"] = (equity_curve(out["ret"]) * AUM_START).shift(1, fill_value=AUM_START)
    out["rf_ec"]  = (equity_curve(out["rf"])  * AUM_START).shift(1, fill_value=AUM_START)
    out["ls_ec"]  = (equity_curve(out["port_ls"]) * AUM_START).shift(1, fill_value=AUM_START)
    return out

for name in ["BASE", "LIQ_ONLY", "ILLIQ_ONLY"]:
    mp = ls_outputs[name]["monthly"]
    if len(mp)==0: 
        continue
    aligned = align_with_market(mp)
    aligned.to_csv(os.path.join(CSV_DIR, f"sp500_vs_our_LS_{name}.csv"), index=False)
    x = range(len(aligned))
    plt.figure(figsize=(14,7))
    plt.plot(x, aligned["ls_ec"],  label=f"Our L–S ({name})", linewidth=2.2)
    plt.plot(x, aligned["mkt_ec"], label="S&P 500", linewidth=2.0, linestyle="--")
    plt.plot(x, aligned["rf_ec"],  label="Risk-free", linewidth=1.8, linestyle=":")
    plt.title(f"S&P500 vs Our L–S — {name}")
    plt.xlabel("Month index (aligned)")
    plt.ylabel(f"Cumulative growth of ${AUM_START:,.0f}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"sp500_vs_our_LS_{name}.png"), dpi=PLOTS_DPI, bbox_inches="tight")
    plt.close()

# C) Liquidity diagnostics (universe) + country removals + CSVs
print("Extracting liquidity diagnostics…")
liq_monthly = (predX.groupby(["year","month"])
               .apply(lambda d: pd.Series({
                   "n_universe": len(d),
                   "n_illiquid": int(d["is_illiquid"].sum()),
                   "pct_illiquid": d["is_illiquid"].mean()
               }))
               .reset_index())
liq_monthly.to_csv(os.path.join(CSV_DIR,"liquidity_universe_monthly.csv"), index=False)

removed = predX[predX["is_illiquid"] == True]
country_removed = (removed.groupby("excntry")["id"].nunique()
                   .sort_values(ascending=False)
                   .rename("n_removed_ids")
                   .reset_index())
country_removed.to_csv(os.path.join(CSV_DIR,"country_illiquid_removed_counts.csv"), index=False)

topN = country_removed.head(15)
plt.figure(figsize=(12,6))
plt.bar(topN["excntry"].astype(str), topN["n_removed_ids"])
plt.title("Illiquid Removals by Country (Top 15, unique ids)")
plt.ylabel("# Unique IDs removed")
plt.xlabel("Country")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"country_illiquid_removed_bar.png"), dpi=PLOTS_DPI, bbox_inches="tight")
plt.close()

# D) Picked (portfolio) illiquidity stats by variant + CSV
def picked_illiquidity_stats(long_df, short_df):
    df = pd.concat([long_df.assign(side="long"), short_df.assign(side="short")], ignore_index=True)
    if df.empty:
        return df.iloc[0:0]
    stats = (df.groupby(["year","month"])
             .apply(lambda d: pd.Series({
                 "n_picked": len(d),
                 "picked_illiquid": int(d["is_illiquid"].sum()),
                 "pct_picked_illiquid": d["is_illiquid"].mean()
             })).reset_index())
    return stats

picked_stats_rows = []
for label, out in ls_outputs.items():
    st = picked_illiquidity_stats(out["long_df"], out["short_df"])
    st["variant"] = label
    picked_stats_rows.append(st)
picked_stats = pd.concat(picked_stats_rows, ignore_index=True) if picked_stats_rows else pd.DataFrame()
picked_stats.to_csv(os.path.join(CSV_DIR,"picked_liquidity_stats_by_variant.csv"), index=False)

liq_monthly["date_key"] = pd.PeriodIndex(pd.to_datetime(liq_monthly["year"].astype(str)+"-"+liq_monthly["month"].astype(str)+"-01"), freq="M")
liq_monthly[["date_key","year","month","n_universe","n_illiquid","pct_illiquid"]].to_csv(
    os.path.join(CSV_DIR,"pct_zero_liquidity_over_time_universe.csv"), index=False
)

plt.figure(figsize=(12,5))
plt.plot(liq_monthly["date_key"].astype(str), liq_monthly["pct_illiquid"], label="% illiquid (universe)", linewidth=2)
if not picked_stats.empty and "LIQ_ONLY" in picked_stats["variant"].unique():
    pk = picked_stats[picked_stats["variant"]=="LIQ_ONLY"].copy()
    pk["date_key"] = pd.PeriodIndex(pd.to_datetime(pk["year"].astype(str)+"-"+pk["month"].astype(str)+"-01"), freq="M")
    pk[["date_key","year","month","pct_picked_illiquid"]].to_csv(
        os.path.join(CSV_DIR,"pct_zero_liquidity_over_time_picked_LIQ_ONLY.csv"), index=False
    )
    plt.plot(pk["date_key"].astype(str), pk["pct_picked_illiquid"], label="% illiquid (picked, LIQ_ONLY)", linewidth=2)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Share")
plt.title("Zero-Liquidity Proxy Share Over Time")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"pct_zero_liquidity_over_time.png"), dpi=PLOTS_DPI, bbox_inches="tight")
plt.close()

# E) Rolling stats (LIQ_ONLY) + CSV
liq_only_monthly = ls_outputs["LIQ_ONLY"]["monthly"].copy().sort_values(["year","month"]).reset_index(drop=True)
liq_only_monthly["time_label"] = liq_only_monthly["year"].astype(str) + "-" + liq_only_monthly["month"].astype(str).str.zfill(2)

liq_only_monthly["rolling_sharpe"] = np.nan
for i in range(len(liq_only_monthly)):
    if i >= 1:
        period = liq_only_monthly["port_ls"].iloc[:i+1]
        if period.std(ddof=1) > 0:
            liq_only_monthly.loc[i, "rolling_sharpe"] = (period.mean() / period.std(ddof=1) * np.sqrt(12))

bm = liq_only_monthly.merge(mkt, on=["year","month"], how="left")
bm["mkt_rf"] = bm["ret"] - bm["rf"]
bm["port_ls_rf"] = bm["port_ls"] - bm["rf"]
liq_only_monthly["rolling_alpha"] = np.nan
for i in range(len(bm)):
    if i >= 3:
        dd = bm.iloc[:i+1].dropna(subset=["port_ls_rf","mkt_rf"])
        try:
            ols = sm.ols("port_ls_rf ~ mkt_rf", data=dd).fit(cov_type="HAC", cov_kwds={"maxlags":3}, use_t=True)
            liq_only_monthly.loc[i, "rolling_alpha"] = ols.params.get("Intercept", np.nan)
        except Exception:
            pass

liq_only_monthly.to_csv(os.path.join(CSV_DIR,"rolling_stats_LIQ_ONLY.csv"), index=False)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
axes[0].bar(range(len(liq_only_monthly)), liq_only_monthly["port_ls"], alpha=0.7)
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].set_xlabel("Month")
axes[0].set_ylabel("L-S Return")
axes[0].set_title("Monthly Long-Short Returns (LIQ_ONLY)")
axes[0].grid(True, alpha=0.3)
step = max(1, len(liq_only_monthly) // 20)
axes[0].set_xticks(range(0, len(liq_only_monthly), step))
axes[0].set_xticklabels(liq_only_monthly["time_label"].iloc[::step], rotation=45, ha='right')

axes[1].plot(range(len(liq_only_monthly)), liq_only_monthly["rolling_sharpe"], linewidth=2)
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Sharpe Ratio")
axes[1].set_title("Rolling Sharpe Ratio (Annualized, Expanding Window) — LIQ_ONLY")
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(range(0, len(liq_only_monthly), step))
axes[1].set_xticklabels(liq_only_monthly["time_label"].iloc[::step], rotation=45, ha='right')

axes[2].plot(range(len(liq_only_monthly)), liq_only_monthly["rolling_alpha"], linewidth=2)
axes[2].axhline(0, color="black", linewidth=0.5)
axes[2].set_xlabel("Month")
axes[2].set_ylabel("Alpha")
axes[2].set_title("Rolling CAPM Alpha (Expanding Window) — LIQ_ONLY")
axes[2].grid(True, alpha=0.3)
axes[2].set_xticks(range(0, len(liq_only_monthly), step))
axes[2].set_xticklabels(liq_only_monthly["time_label"].iloc[::step], rotation=45, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"performance_metrics_over_time.png"), dpi=PLOTS_DPI, bbox_inches='tight')
plt.close()

# F) Top 10 long/short holdings by cumulative return (LIQ_ONLY) + CSVs
liq_long  = ls_outputs["LIQ_ONLY"]["long_df"]
liq_short = ls_outputs["LIQ_ONLY"]["short_df"]

long_holdings_cum = liq_long.groupby("id").agg({
    "stock_ret": lambda x: (1 + x).prod() - 1,
    "date": "count" if "date" in liq_long.columns else "size"
}).reset_index()
long_holdings_cum.columns = ["id", "cumulative_return", "months_held"]
top_10_longs = long_holdings_cum.nlargest(10, "cumulative_return")
top_10_longs.to_csv(os.path.join(CSV_DIR,"top_10_long_holdings_LIQ_ONLY.csv"), index=False)

short_holdings_cum = liq_short.groupby("id").agg({
    "stock_ret": lambda x: (1 + x).prod() - 1,
    "date": "count" if "date" in liq_short.columns else "size"
}).reset_index()
short_holdings_cum.columns = ["id", "cumulative_return", "months_held"]
top_10_shorts = short_holdings_cum.nsmallest(10, "cumulative_return")
top_10_shorts.to_csv(os.path.join(CSV_DIR,"top_10_short_holdings_LIQ_ONLY.csv"), index=False)

print("\n=== Summary outputs ===")
print("CSV folder (./csv):")
print(" - summary_stats_by_variant.csv        <-- one row per variant (BASE, LIQ_ONLY, ILLIQ_ONLY) + avg_*_turnover")
print(" - monthly_ls_<VARIANT>.csv            <-- per-variant monthly L/S series (returns + counts + turnover stats)")
print(" - changes_<VARIANT>.csv               <-- per-variant per-month change log (added/kept/removed by side)")
print(" - ls_equity_liquid_vs_illiquid_vs_base.csv")
print(" - sp500_vs_our_LS_<VARIANT>.csv       (BASE, LIQ_ONLY, ILLIQ_ONLY)")
print(" - liquidity_universe_monthly.csv")
print(" - country_illiquid_removed_counts.csv")
print(" - picked_liquidity_stats_by_variant.csv")
print(" - pct_zero_liquidity_over_time_universe.csv")
print(" - pct_zero_liquidity_over_time_picked_LIQ_ONLY.csv")
print(" - rolling_stats_LIQ_ONLY.csv")
print(" - top_10_long_holdings_LIQ_ONLY.csv, top_10_short_holdings_LIQ_ONLY.csv")
print("FIGS folder (./figs):")
print(" - ls_equity_liquid_vs_illiquid_vs_base.png")
print(" - sp500_vs_our_LS_BASE.png, sp500_vs_our_LS_LIQ_ONLY.png, sp500_vs_our_LS_ILLIQ_ONLY.png")
print(" - country_illiquid_removed_bar.png")
print(" - pct_zero_liquidity_over_time.png")
print(" - performance_metrics_over_time.png")
print("\nDone.")
