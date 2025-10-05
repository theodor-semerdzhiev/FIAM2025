import pandas as pd
import numpy as np
import math
import statsmodels.formula.api as sm
from pandas.tseries.offsets import MonthBegin
import matplotlib.pyplot as plt

# ------------------------------
# Config
# ------------------------------
PRED_PATH = "output.csv"
MODEL_COL = "blend"            # prediction column to use
TOP_EACH_CAP = 125             # max names on each side
MIN_TOTAL = 100                # min total names (will try to split ≈50/50)
PERCENTILE = 1                 # use top/bottom PERCENTILE% as initial threshold

# ------------------------------
# Load predictions (must include: date, year, month, id, stock_ret, and MODEL_COL)
# ------------------------------
print("uheloo")
pred = pd.read_csv(PRED_PATH, parse_dates=["date"])

# Safety: ensure year/month exist (derive from 'date' if needed)
if not {"year","month"}.issubset(pred.columns):
    pred["year"]  = pred["date"].dt.year
    pred["month"] = pred["date"].dt.month

# ------------------------------
# Helper: select long/short positions per (year, month)
# ------------------------------
def select_portfolios_one_month(df_month: pd.DataFrame,
                                model_col: str,
                                top_each_cap: int,
                                min_total: int,
                                percentile: float):
    """
    df_month: rows for a single (year, month)
    Returns two DataFrames: longs_df, shorts_df (subsets of df_month)
    """
    n = len(df_month)
    if n == 0:
        return df_month.iloc[0:0], df_month.iloc[0:0]

    # You cannot hold more than half the universe per side
    k_cap = min(top_each_cap, n // 2)

    # Compute percentile thresholds
    q_long  = df_month[model_col].quantile(1 - percentile/100.0)
    q_short = df_month[model_col].quantile(percentile/100.0)

    # Initial threshold selections
    longs_init  = df_month[df_month[model_col] >= q_long].copy()
    shorts_init = df_month[df_month[model_col] <= q_short].copy()

    # Rank within month for deterministic tie-breaking
    # High scores first for longs; low scores first for shorts.
    longs_init  = longs_init.sort_values([model_col, "id"], ascending=[False, True])
    shorts_init = shorts_init.sort_values([model_col, "id"], ascending=[True, True])

    # Trim to cap if threshold pulls too many
    longs_sel  = longs_init.head(k_cap)
    shorts_sel = shorts_init.head(k_cap)

    # If too few names in total, fill from next-best beyond threshold (still respecting caps)
    total_now = len(longs_sel) + len(shorts_sel)
    if total_now < min_total:
        deficit = min_total - total_now
        # Remaining candidates after current selections
        chosen_ids = set(pd.concat([longs_sel[["id"]], shorts_sel[["id"]]])["id"].tolist())

        longs_remaining = (df_month[~df_month["id"].isin(chosen_ids)]
                           .sort_values([model_col, "id"], ascending=[False, True]))
        shorts_remaining = (df_month[~df_month["id"].isin(chosen_ids)]
                            .sort_values([model_col, "id"], ascending=[True, True]))

        # Try to split the deficit ~half/half
        add_longs_target = min(k_cap - len(longs_sel), math.ceil(deficit / 2))
        add_shorts_target = min(k_cap - len(shorts_sel), deficit - add_longs_target)

        # Add longs
        if add_longs_target > 0:
            to_add = longs_remaining.head(add_longs_target)
            longs_sel = pd.concat([longs_sel, to_add], ignore_index=True)
            chosen_ids.update(to_add["id"].tolist())

        # Add shorts
        if add_shorts_target > 0:
            # refresh remaining after possibly adding longs
            shorts_remaining = (df_month[~df_month["id"].isin(chosen_ids)]
                                .sort_values([model_col, "id"], ascending=[True, True]))
            to_add = shorts_remaining.head(add_shorts_target)
            shorts_sel = pd.concat([shorts_sel, to_add], ignore_index=True)
            chosen_ids.update(to_add["id"].tolist())

        # If one side couldn't be filled, try to put the leftover on the other side (still <= cap)
        total_now = len(longs_sel) + len(shorts_sel)
        if total_now < min_total:
            leftover = min_total - total_now
            # Try longs first
            if len(longs_sel) < k_cap and leftover > 0:
                longs_remaining = (df_month[~df_month["id"].isin(chosen_ids)]
                                   .sort_values([model_col, "id"], ascending=[False, True]))
                to_add = longs_remaining.head(min(leftover, k_cap - len(longs_sel)))
                longs_sel = pd.concat([longs_sel, to_add], ignore_index=True)
                chosen_ids.update(to_add["id"].tolist())
                leftover = min_total - (len(longs_sel) + len(shorts_sel))
            # Then shorts
            if len(shorts_sel) < k_cap and leftover > 0:
                shorts_remaining = (df_month[~df_month["id"].isin(chosen_ids)]
                                    .sort_values([model_col, "id"], ascending=[True, True]))
                to_add = shorts_remaining.head(min(leftover, k_cap - len(shorts_sel)))
                shorts_sel = pd.concat([shorts_sel, to_add], ignore_index=True)
                chosen_ids.update(to_add["id"].tolist())

    # Final safety caps (never exceed k_cap each)
    longs_sel  = longs_sel.head(k_cap)
    shorts_sel = shorts_sel.head(k_cap)

    return longs_sel, shorts_sel

# ------------------------------
# Build monthly long/short memberships
# ------------------------------
groups = pred.groupby(["year", "month"], sort=True, as_index=False)

long_rows = []
short_rows = []

for (y, m), dfm in groups:
    ldf, sdf = select_portfolios_one_month(dfm, MODEL_COL, TOP_EACH_CAP, MIN_TOTAL, PERCENTILE)
    ldf = ldf.assign(year=y, month=m, side="long")
    sdf = sdf.assign(year=y, month=m, side="short")
    long_rows.append(ldf)
    short_rows.append(sdf)

long_df  = pd.concat(long_rows,  ignore_index=True) if long_rows  else pred.iloc[0:0]
short_df = pd.concat(short_rows, ignore_index=True) if short_rows else pred.iloc[0:0]

# ------------------------------
# Compute monthly equal-weight returns
# ------------------------------
long_ret = (long_df.groupby(["year","month"])["stock_ret"].mean()
            .rename("long_ret"))
short_ret = (short_df.groupby(["year","month"])["stock_ret"].mean()
             .rename("short_ret"))

monthly_port = pd.concat([long_ret, short_ret], axis=1).dropna().reset_index()
monthly_port["port_ls"] = monthly_port["long_ret"] - monthly_port["short_ret"]

# For reference: how many names each month on each side and total
counts_long  = long_df.groupby(["year","month"])["id"].nunique().rename("n_long")
counts_short = short_df.groupby(["year","month"])["id"].nunique().rename("n_short")
monthly_port = (monthly_port
                .merge(counts_long,  on=["year","month"], how="left")
                .merge(counts_short, on=["year","month"], how="left"))
monthly_port["n_total"] = monthly_port["n_long"] + monthly_port["n_short"]

# ------------------------------
# Performance stats (LS)
# ------------------------------
# Sharpe (annualized, monthly data)
if monthly_port["port_ls"].std(ddof=1) > 0:
    sharpe = monthly_port["port_ls"].mean() / monthly_port["port_ls"].std(ddof=1) * np.sqrt(12)
else:
    sharpe = np.nan
print("Sharpe Ratio (L-S):", sharpe)

# CAPM Alpha (LS) vs market
mkt = pd.read_csv("data/mkt_ind.csv")  # expected columns: year, month, ret, rf
monthly_port = monthly_port.merge(mkt, how="inner", on=["year", "month"])
monthly_port["mkt_rf"] = monthly_port["ret"] - monthly_port["rf"]
# Create excess return for the long-short portfolio
monthly_port["port_ls_rf"] = monthly_port["port_ls"] - monthly_port["rf"]

# Now regress excess returns on both sides
nw_ols = sm.ols("port_ls_rf ~ mkt_rf", data=monthly_port).fit(
    cov_type="HAC", cov_kwds={"maxlags": 3}, use_t=True
)
print(nw_ols.summary())
print("CAPM Alpha:", nw_ols.params.get("Intercept", np.nan))
print("t-statistic:", nw_ols.tvalues.get("Intercept", np.nan))
if nw_ols.mse_resid > 0:
    info_ratio = nw_ols.params.get("Intercept", np.nan) / np.sqrt(nw_ols.mse_resid) * np.sqrt(12)
else:
    info_ratio = np.nan
print("Information Ratio (annualized):", info_ratio)

# Max 1-month loss (LS)
max_1m_loss = monthly_port["port_ls"].min()
print("Max 1-Month Loss (L-S):", max_1m_loss)

# Max Drawdown (LS) using log-cum
monthly_port["log_ls"] = np.log1p(monthly_port["port_ls"])
monthly_port["cum_log_ls"] = monthly_port["log_ls"].cumsum()
rolling_peak = monthly_port["cum_log_ls"].cummax()
drawdowns = rolling_peak - monthly_port["cum_log_ls"]
max_drawdown = drawdowns.max()
print("Maximum Drawdown (log scale):", max_drawdown)

# ------------------------------
# Turnover (by membership change) for each side
# ------------------------------
def turnover_count(df_side: pd.DataFrame) -> float:
    """
    df_side: rows for a single side (long or short) across months with columns ['id','date']
    Measures average % replaced month-to-month (1 - overlap/previous_count).
    """
    if df_side.empty:
        return np.nan

    # Membership at *start* of each month (normalize date to month start)
    side = df_side[["id","date"]].copy()
    side["month_start"] = (side["date"] - MonthBegin(1)).dt.normalize() + MonthBegin(1)
    # For safety if 'date' is already at month end etc.
    side["month_start"] = side["month_start"].dt.to_period("M").dt.to_timestamp()

    # Unique membership by month
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

long_tov  = turnover_count(long_df)
short_tov = turnover_count(short_df)
print("Long Portfolio Turnover:", long_tov)
print("Short Portfolio Turnover:", short_tov)

# ------------------------------
# Quick sanity prints
# ------------------------------
print("\nSample monthly counts")
print(monthly_port[["year","month","n_long","n_short","n_total"]])

# ------------------------------
# NEW: Generate plots of returns, Alpha, and Sharpe at different points in time
# ------------------------------
# Create time labels for x-axis
monthly_port["time_label"] = monthly_port["year"].astype(str) + "-" + monthly_port["month"].astype(str).str.zfill(2)

# Calculate rolling Sharpe ratio (using expanding window)
monthly_port["rolling_sharpe"] = np.nan
for i in range(len(monthly_port)):
    if i >= 1:  # Need at least 2 months
        period_returns = monthly_port["port_ls"].iloc[:i+1]
        if period_returns.std(ddof=1) > 0:
            monthly_port.loc[monthly_port.index[i], "rolling_sharpe"] = (
                period_returns.mean() / period_returns.std(ddof=1) * np.sqrt(12)
            )

# Calculate rolling Alpha (using expanding window)
monthly_port["rolling_alpha"] = np.nan
for i in range(len(monthly_port)):
    if i >= 3:  # Need at least 4 months for regression
        period_data = monthly_port.iloc[:i+1]
        try:
            period_ols = sm.ols("port_ls_rf ~ mkt_rf", data=period_data).fit(
                cov_type="HAC", cov_kwds={"maxlags": 3}, use_t=True
            )
            monthly_port.loc[monthly_port.index[i], "rolling_alpha"] = period_ols.params.get("Intercept", np.nan)
        except:
            pass

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Monthly Returns
axes[0].bar(range(len(monthly_port)), monthly_port["port_ls"], color='steelblue', alpha=0.7)
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[0].set_xlabel("Month")
axes[0].set_ylabel("L-S Return")
axes[0].set_title("Monthly Long-Short Returns")
axes[0].grid(True, alpha=0.3)
# Set x-axis labels (show every nth label to avoid crowding)
step = max(1, len(monthly_port) // 20)
axes[0].set_xticks(range(0, len(monthly_port), step))
axes[0].set_xticklabels(monthly_port["time_label"].iloc[::step], rotation=45, ha='right')

# Plot 2: Rolling Sharpe Ratio
axes[1].plot(range(len(monthly_port)), monthly_port["rolling_sharpe"], color='green', linewidth=2)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Sharpe Ratio")
axes[1].set_title("Rolling Sharpe Ratio (Annualized, Expanding Window)")
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(range(0, len(monthly_port), step))
axes[1].set_xticklabels(monthly_port["time_label"].iloc[::step], rotation=45, ha='right')

# Plot 3: Rolling Alpha
axes[2].plot(range(len(monthly_port)), monthly_port["rolling_alpha"], color='red', linewidth=2)
axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[2].set_xlabel("Month")
axes[2].set_ylabel("Alpha")
axes[2].set_title("Rolling CAPM Alpha (Expanding Window)")
axes[2].grid(True, alpha=0.3)
axes[2].set_xticks(range(0, len(monthly_port), step))
axes[2].set_xticklabels(monthly_port["time_label"].iloc[::step], rotation=45, ha='right')

plt.tight_layout()
plt.savefig("performance_metrics_over_time.png", dpi=300, bbox_inches='tight')
print("\nSaved plot: performance_metrics_over_time.png")
plt.close()

# ------------------------------
# NEW: Extract top 10 long holdings based on cumulative returns
# ------------------------------
# Calculate cumulative return for each stock while it was held long
long_holdings_cum = long_df.groupby("id").agg({
    "stock_ret": lambda x: (1 + x).prod() - 1,  # Cumulative return
    "date": "count"  # Number of months held
}).reset_index()
long_holdings_cum.columns = ["id", "cumulative_return", "months_held"]

# Sort by cumulative return and get top 10
top_10_longs = long_holdings_cum.nlargest(10, "cumulative_return")

print("\n" + "="*60)
print("TOP 10 LONG HOLDINGS BY CUMULATIVE RETURN")
print("="*60)
print(top_10_longs.to_string(index=False))
print("="*60)

# Save to CSV
top_10_longs.to_csv("top_10_long_holdings.csv", index=False)
print("\nSaved top 10 long holdings to: top_10_long_holdings.csv")

# ------------------------------
# NEW: Extract top 10 short holdings based on cumulative returns
# ------------------------------
# Calculate cumulative return for each stock while it was held short
short_holdings_cum = short_df.groupby("id").agg({
    "stock_ret": lambda x: (1 + x).prod() - 1,  # Cumulative return
    "date": "count"  # Number of months held
}).reset_index()
short_holdings_cum.columns = ["id", "cumulative_return", "months_held"]

# Sort by cumulative return and get top 10 (lowest/most negative returns are best for shorts)
top_10_shorts = short_holdings_cum.nsmallest(10, "cumulative_return")

print("\n" + "="*60)
print("TOP 10 SHORT HOLDINGS BY CUMULATIVE RETURN (Most Negative)")
print("="*60)
print(top_10_shorts.to_string(index=False))
print("="*60)

# Save to CSV
top_10_shorts.to_csv("top_10_short_holdings.csv", index=False)
print("\nSaved top 10 short holdings to: top_10_short_holdings.csv")