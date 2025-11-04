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
from pandas.tseries.offsets import MonthEnd

# ------------------------------
# Config
# ------------------------------
PRED_PATH     = "output_final.csv"          # predictions file (must include: date, year, month, id, stock_ret, and MODEL_COL)
FEATURES_PATH = "data/ret_sample.parquet"   # original features universe (for liquidity proxies, country, iid, etc.)
MKT_PATH      = "data/mkt_ind.csv"          # market data (rf, ret) monthly

MODEL_COL     = "blend"               # prediction column to use
TOP_EACH_CAP  = 125                   # max names on each side
MIN_TOTAL     = 100                   # min total names (≈50/50 if possible)
PERCENTILE    = 1                     # top/bottom PERCENTILE% as initial threshold

# ==============================
# STRICT Liquidity rules (time-t)
# ==============================
# Philosophy:
#  - Treat *any* missing proxy as illiquid (strict).
#  - Zero-trade fractions: low tolerance (>= 10%) over any horizon ⇒ illiquid.
#  - Dollar volume & turnover: use *monthly cross-sectional* low-quantile cutoffs (e.g., bottom 30%) ⇒ illiquid.
#  - Bid-ask (21d HL proxy): use *monthly cross-sectional* high-quantile cutoff (e.g., top 70%) ⇒ illiquid.
#  - Also keep absolute "zero" checks (==0) for volume/turnover.
#
# If you want to dial strictness up/down, tweak the *_Q* quantiles or absolute thresholds below.
STRICT_ZERO_TRADE_THRESH = {
    "zero_trades_21d": 0.2,    # >= 10% zero-trade days in last 21d ⇒ illiquid
    "zero_trades_126d": 0.3,   # >= 10% over 6m
    "zero_trades_252d": 0.3    # >= 10% over 12m
}

# Cross-sectional quantiles (computed each char-month t on the *feature* file)
DOLVOL_LOW_Q     = 0.20   # bottom 30% dollar volume ⇒ illiquid
TURNOVER_LOW_Q   = 0.20   # bottom 30% turnover ⇒ illiquid
BIDASK_HIGH_Q    = 0.80   # top 30% (>= 70th pct) bid-ask HL ⇒ illiquid

# Optional absolute clamps (kept strict but scale-free). Leave as None to skip.
# (Because units can vary by dataset, we rely primarily on quantiles; zeros always fail.)
ABS_MIN_PRICE    = 0.00   # < $2 ⇒ illiquid (set None to disable)
ABS_MAX_BIDASKHL = 0.25   # if bidaskhl_21d > 0.25 ⇒ illiquid (set None to disable)

# Column names (must match FEATURES_PATH columns if present)
LIQ_COLS = {
    "zero_trades_21d",
    "zero_trades_126d",
    "zero_trades_252d",
    "dolvol_126d",
    "turnover_126d"
}
BIDASK_COL = "bidaskhl_21d"   # optional
PRICE_COL  = "prc"            # optional penny stock filter

# ---- Concentration + turnover caps ----
MAX_COUNTRY_WEIGHT = 0.75   # no country > 75% of long AUM or short AUM
MIN_COUNTRIES_PER_SIDE = 2  # enforce >= 2 countries represented per side
TURNOVER_MAX_LONG  = 0.45   # at most 50% of long book can turn over month-to-month
TURNOVER_MAX_SHORT = 0.45   # at most 50% of short book can turn over month-to-month

# ---- Constant 70/30 exposure weights for the L/S strategy ----
LONG_WEIGHT  = 0.70         # 70% long notional weight
SHORT_WEIGHT = 0.30         # 30% short notional weight

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
    """Simple to cumulative."""
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
    # annualized Info Ratio from monthly alpha / residual std
    if getattr(ols, "mse_resid", None) is not None and ols.mse_resid > 0:
        ir_ann = alpha / np.sqrt(ols.mse_resid) * np.sqrt(12)
    else:
        ir_ann = np.nan
    return {"alpha": alpha, "alpha_t": tstat, "ir_annual": float(ir_ann)}

def compute_summary_stats(mp: pd.DataFrame, mkt_df: pd.DataFrame, label: str) -> dict:
    """One summary row per variant."""
    row = {"variant": label}
    if mp.empty:
        # fill with NaNs
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
    """
    Compute per (char_year,char_month) cutoffs used for strict screening.
    Uses only 't' features so there is no look-ahead.
    """
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
    # Missing any key proxy ⇒ illiquid (strict stance)
    needed_cols = set(LIQ_COLS) | ({BIDASK_COL} if BIDASK_COL else set()) | ({PRICE_COL} if PRICE_COL else set())
    for c in needed_cols:
        if c and c in row and pd.isna(row[c]):
            return True

    # 1) Zero-trade fractions: any horizon >= threshold ⇒ illiquid
    for c, thr in STRICT_ZERO_TRADE_THRESH.items():
        if c in row and pd.notna(row[c]) and float(row[c]) >= thr:
            return True

    # 2) Dollar volume + turnover:
    #    - absolute zero ⇒ illiquid
    #    - below (t) cross-sectional low-quantile ⇒ illiquid
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

    # 3) Bid-ask HL: above cross-sectional high-quantile OR above absolute cap ⇒ illiquid
    if BIDASK_COL and BIDASK_COL in row and pd.notna(row[BIDASK_COL]):
        ba = float(row[BIDASK_COL])
        q_hi = row.get("q_bidask_high", np.nan)
        if pd.notna(q_hi) and ba >= q_hi:
            return True
        if ABS_MAX_BIDASKHL is not None and ba > ABS_MAX_BIDASKHL:
            return True
    elif BIDASK_COL:
        # missing bid-ask ⇒ illiquid (strict)
        return True

    # 4) Price floor (optional, strict)
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
                                top_each_cap: int,
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

    # Defaults for new knobs if not provided
    if turnover_max_long is None:  turnover_max_long  = TURNOVER_MAX_LONG
    if turnover_max_short is None: turnover_max_short = TURNOVER_MAX_SHORT
    if max_country_weight is None: max_country_weight = MAX_COUNTRY_WEIGHT
    if min_countries_per_side is None: min_countries_per_side = MIN_COUNTRIES_PER_SIDE
    if prev_long_ids is None:  prev_long_ids  = set()
    if prev_short_ids is None: prev_short_ids = set()

    # Helper: ranked candidate lists (by signal) for each side
    def ranked(df, side):
        if side == "long":
            return df.sort_values([model_col,"id"], ascending=[False,True]).copy()
        else:  # short
            return df.sort_values([model_col,"id"], ascending=[True,True]).copy()

    # Candidate pools by percentile threshold
    q_long  = df_month[model_col].quantile(1 - percentile/100.0)
    q_short = df_month[model_col].quantile(percentile/100.0)

    longs_init_candidates  = df_month[df_month[model_col] >= q_long].copy()
    shorts_init_candidates = df_month[df_month[model_col] <= q_short].copy()

    longs_init  = ranked(longs_init_candidates, "long")
    shorts_init = ranked(shorts_init_candidates, "short")

    # --- explicit 70% long / 30% short name targets ---
    avail_L = len(longs_init)
    avail_S = len(shorts_init)
    cap_L = min(top_each_cap, avail_L)
    cap_S = min(top_each_cap, avail_S)

    def max_total_allowed(cL, cS):
        m1 = int(math.floor(cL / 0.70)) if cL > 0 else 0
        m2 = int(math.floor(cS / 0.30)) if cS > 0 else 0
        return min(m1, m2)

    max_total = min(n, cap_L + cap_S, max_total_allowed(cap_L, cap_S))
    total_target = min(max_total, max(min_total, 0))
    if total_target <= 0:
        return df_month.iloc[0:0], df_month.iloc[0:0]

    k_long_target  = int(round(0.70 * total_target))
    k_short_target = total_target - k_long_target

    # Helper: country-aware picker with turnover preference
    def pick_with_constraints(ranked_df, k_target, side, prev_ids, turnover_cap, max_ctry_wt, min_ctrys):
        rd = ranked_df.copy()
        if "excntry" not in rd.columns:
            rd["excntry"] = "UNK"
        rd["excntry"] = rd["excntry"].astype(str).fillna("UNK")

        allowed_changes = int(math.floor(turnover_cap * k_target))
        min_keep = max(0, k_target - allowed_changes)

        rd_prev = rd[rd["id"].isin(prev_ids)].copy()
        rd_new  = rd[~rd["id"].isin(prev_ids)].copy()

        max_count = max(1, int(math.floor(max_ctry_wt * k_target)))

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

        # Pass 1: keep previous holdings up to min_keep while respecting caps
        for _, row in rd_prev.iterrows():
            if len(selected_rows) >= min_keep:
                break
            try_add(row)

        # Pass 2: fill remaining slots
        for _, row in pd.concat([rd_prev, rd_new], ignore_index=True).iterrows():
            if len(selected_rows) >= k_target:
                break
            try_add(row)

        # Relax country cap if needed
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

        # Enforce minimum number of countries (swap if needed)
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

        # Rebalance if over country cap due to swaps
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

        # Final size enforcement
        if len(sel) > k_target:
            if side == "long":
                sel = sel.sort_values([model_col,"id"], ascending=[False,True]).head(k_target)
            else:
                sel = sel.sort_values([model_col,"id"], ascending=[True,True]).head(k_target)
        return sel.reset_index(drop=True)

    longs_sel  = pick_with_constraints(longs_init,  k_long_target,  "long",  prev_long_ids,  TURNOVER_MAX_LONG,  MAX_COUNTRY_WEIGHT, MIN_COUNTRIES_PER_SIDE)
    shorts_sel = pick_with_constraints(shorts_init, k_short_target, "short", prev_short_ids, TURNOVER_MAX_SHORT, MAX_COUNTRY_WEIGHT, MIN_COUNTRIES_PER_SIDE)

    # Preserve 70/30 ratio if constraints cut availability
    final_L = len(longs_sel)
    final_S = len(shorts_sel)
    if final_L + final_S > 0:
        max_total_given_picks = min(int(math.floor(final_L / 0.70)), int(math.floor(final_S / 0.30)))
        if max_total_given_picks > 0:
            want_L = int(round(0.70 * max_total_given_picks))
            want_S = max_total_given_picks - want_L
            if final_L > want_L:
                longs_sel = longs_sel.sort_values([MODEL_COL,"id"], ascending=[False,True]).head(want_L)
            if final_S > want_S:
                shorts_sel = shorts_sel.sort_values([MODEL_COL,"id"], ascending=[True,True]).head(want_S)

    return longs_sel.reset_index(drop=True), shorts_sel.reset_index(drop=True)

def build_long_short(pred_like: pd.DataFrame, label: str):
    groups = pred_like.groupby(["year","month"], sort=True, as_index=False)
    long_rows, short_rows = [], []
    prev_long_ids, prev_short_ids = set(), set()
    for (y, m), dfm in groups:
        ldf, sdf = select_portfolios_one_month(
            dfm, MODEL_COL, TOP_EACH_CAP, MIN_TOTAL, PERCENTILE,
            prev_long_ids=prev_long_ids,
            prev_short_ids=prev_short_ids,
            turnover_max_long=TURNOVER_MAX_LONG,
            turnover_max_short=TURNOVER_MAX_SHORT,
            max_country_weight=MAX_COUNTRY_WEIGHT,
            min_countries_per_side=MIN_COUNTRIES_PER_SIDE
        )
        ldf = ldf.assign(year=y, month=m, side="long", variant=label)
        sdf = sdf.assign(year=y, month=m, side="short", variant=label)
        long_rows.append(ldf)
        short_rows.append(sdf)
        prev_long_ids = set(ldf["id"].tolist())
        prev_short_ids = set(sdf["id"].tolist())

    long_df  = pd.concat(long_rows,  ignore_index=True) if long_rows  else pred_like.iloc[0:0]
    short_df = pd.concat(short_rows, ignore_index=True) if short_rows else pred_like.iloc[0:0]

    long_ret  = long_df.groupby(["year","month"])["stock_ret"].mean().rename("long_ret")
    short_ret = short_df.groupby(["year","month"])["stock_ret"].mean().rename("short_ret")

    monthly = pd.concat([long_ret, short_ret], axis=1).dropna().reset_index()
    monthly["port_ls"] = LONG_WEIGHT * monthly["long_ret"] - SHORT_WEIGHT * monthly["short_ret"]
    counts_long  = long_df.groupby(["year","month"])["id"].nunique().rename("n_long")
    counts_short = short_df.groupby(["year","month"])["id"].nunique().rename("n_short")
    monthly = (monthly.merge(counts_long,  on=["year","month"], how="left")
                      .merge(counts_short, on=["year","month"], how="left"))
    monthly["n_total"] = monthly["n_long"] + monthly["n_short"]
    monthly["variant"] = label
    return long_df, short_df, monthly

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
    long_df, short_df, monthly = build_long_short(uni, label)
    ls_outputs[label] = {"long_df": long_df, "short_df": short_df, "monthly": monthly}
    if not monthly.empty:
        outp = os.path.join(CSV_DIR, f"monthly_ls_{label}.csv")
        monthly.sort_values(["year","month"]).to_csv(outp, index=False)

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
    summary_rows.append(compute_summary_stats(mp, mkt, label))

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
    # Shift so first plotted point is exactly 1.0
    mp["equity_ls"] = equity_curve(mp["port_ls"]).shift(1, fill_value=1.0)
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
    # Use shifted equity so every series starts at 1.0
    ec = equity_curve(mp.sort_values(["year","month"])["port_ls"]).shift(1, fill_value=1.0)
    plt.plot(range(len(ec)), ec, label=key, linewidth=2)
plt.ylabel("Equity (L–S, $1 start)")
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
    # Shift all cumulative lines so the first point is 1.0
    out["mkt_ec"] = equity_curve(out["ret"]).shift(1, fill_value=1.0)
    out["rf_ec"]  = equity_curve(out["rf"]).shift(1, fill_value=1.0)
    out["ls_ec"]  = equity_curve(out["port_ls"]).shift(1, fill_value=1.0)
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
    plt.ylabel("Cumulative growth of $1")
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

# Plot: Country removals (illiquid)
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

# Plot & CSV: % zero-liquidity over time (universe + picked LIQ_ONLY)
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

# Rolling Sharpe (expanding)
liq_only_monthly["rolling_sharpe"] = np.nan
for i in range(len(liq_only_monthly)):
    if i >= 1:
        period = liq_only_monthly["port_ls"].iloc[:i+1]
        if period.std(ddof=1) > 0:
            liq_only_monthly.loc[i, "rolling_sharpe"] = (period.mean() / period.std(ddof=1) * np.sqrt(12))

# Rolling alpha vs market (expanding, HAC on each slice)
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

# Save CSV behind the rolling plots
liq_only_monthly.to_csv(os.path.join(CSV_DIR,"rolling_stats_LIQ_ONLY.csv"), index=False)

# Plots
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
print(" - summary_stats_by_variant.csv        <-- one row per variant (BASE, LIQ_ONLY, ILLIQ_ONLY)")
print(" - monthly_ls_<VARIANT>.csv            <-- per-variant monthly L/S series (returns + counts)")
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
