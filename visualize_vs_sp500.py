#!/usr/bin/env python3
"""
Generate comparison visualization: Optimized Strategy vs S&P 500
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load data
strategy_df = pd.read_csv("csv/monthly_ls_OPTIMIZED.csv")
mkt_df = pd.read_csv("fiamdata/mkt_ind.csv")

# Merge on year/month
merged = strategy_df.merge(mkt_df[['year', 'month', 'ret', 'rf']], on=['year', 'month'], how='left')
merged['date'] = pd.to_datetime(merged[['year', 'month']].assign(day=1))

# Calculate cumulative returns
merged['cum_strategy'] = (1 + merged['port_ls']).cumprod()
merged['cum_sp500'] = (1 + merged['ret']).cumprod()

# Calculate metrics
def calc_metrics(returns, rf):
    excess = returns - rf
    sharpe = (excess.mean() / returns.std()) * np.sqrt(12)
    cum_ret = (1 + returns).cumprod()
    n_years = len(returns) / 12
    cagr = (cum_ret.iloc[-1] ** (1/n_years) - 1) * 100

    # Max drawdown
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    max_dd = drawdown.min() * 100

    # Volatility
    vol = returns.std() * np.sqrt(12) * 100

    # Hit rate
    hit_rate = (returns > 0).mean() * 100

    return {
        'sharpe': sharpe,
        'cagr': cagr,
        'max_dd': max_dd,
        'vol': vol,
        'hit_rate': hit_rate
    }

strat_metrics = calc_metrics(merged['port_ls'], merged['rf'].fillna(0))
sp500_metrics = calc_metrics(merged['ret'], merged['rf'].fillna(0))

# Create figure
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

# 1. Cumulative Returns
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(merged['date'], merged['cum_strategy'] * 100, label='Optimized Strategy', linewidth=2, color='#2E86DE')
ax1.plot(merged['date'], merged['cum_sp500'] * 100, label='S&P 500', linewidth=2, color='#EE5A24', alpha=0.8)
ax1.set_title('Cumulative Returns: Optimized Strategy vs S&P 500', fontsize=14, fontweight='bold')
ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('Date', fontsize=11)

# 2. Rolling 12-Month Sharpe Ratio
ax2 = fig.add_subplot(gs[1, 0])
merged['rolling_sharpe_strat'] = merged['port_ls'].rolling(12).apply(
    lambda x: (x.mean() / x.std()) * np.sqrt(12) if len(x) == 12 else np.nan
)
merged['rolling_sharpe_sp500'] = merged['ret'].rolling(12).apply(
    lambda x: (x.mean() / x.std()) * np.sqrt(12) if len(x) == 12 else np.nan
)
ax2.plot(merged['date'], merged['rolling_sharpe_strat'], label='Strategy', linewidth=1.5, color='#2E86DE')
ax2.plot(merged['date'], merged['rolling_sharpe_sp500'], label='S&P 500', linewidth=1.5, color='#EE5A24', alpha=0.8)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_title('Rolling 12-Month Sharpe Ratio', fontsize=12, fontweight='bold')
ax2.set_ylabel('Sharpe Ratio', fontsize=10)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. Drawdown Comparison
ax3 = fig.add_subplot(gs[1, 1])
cum_strat = (1 + merged['port_ls']).cumprod()
running_max_strat = cum_strat.cummax()
dd_strat = ((cum_strat - running_max_strat) / running_max_strat) * 100

cum_sp500 = (1 + merged['ret']).cumprod()
running_max_sp500 = cum_sp500.cummax()
dd_sp500 = ((cum_sp500 - running_max_sp500) / running_max_sp500) * 100

ax3.fill_between(merged['date'], dd_strat, 0, alpha=0.5, color='#2E86DE', label='Strategy')
ax3.fill_between(merged['date'], dd_sp500, 0, alpha=0.3, color='#EE5A24', label='S&P 500')
ax3.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
ax3.set_ylabel('Drawdown (%)', fontsize=10)
ax3.legend(loc='lower left', fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. Monthly Return Distribution
ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(merged['port_ls'] * 100, bins=40, alpha=0.6, color='#2E86DE', label='Strategy', edgecolor='black')
ax4.hist(merged['ret'] * 100, bins=40, alpha=0.4, color='#EE5A24', label='S&P 500', edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax4.set_title('Monthly Return Distribution', fontsize=12, fontweight='bold')
ax4.set_xlabel('Monthly Return (%)', fontsize=10)
ax4.set_ylabel('Frequency', fontsize=10)
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

# 5. Performance Metrics Table
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

metrics_data = [
    ['Metric', 'Strategy', 'S&P 500', 'Difference'],
    ['Sharpe Ratio', f"{strat_metrics['sharpe']:.3f}", f"{sp500_metrics['sharpe']:.3f}",
     f"{strat_metrics['sharpe'] - sp500_metrics['sharpe']:+.3f}"],
    ['CAGR', f"{strat_metrics['cagr']:.2f}%", f"{sp500_metrics['cagr']:.2f}%",
     f"{strat_metrics['cagr'] - sp500_metrics['cagr']:+.2f}%"],
    ['Volatility', f"{strat_metrics['vol']:.2f}%", f"{sp500_metrics['vol']:.2f}%",
     f"{strat_metrics['vol'] - sp500_metrics['vol']:+.2f}%"],
    ['Max Drawdown', f"{strat_metrics['max_dd']:.2f}%", f"{sp500_metrics['max_dd']:.2f}%",
     f"{strat_metrics['max_dd'] - sp500_metrics['max_dd']:+.2f}%"],
    ['Hit Rate', f"{strat_metrics['hit_rate']:.1f}%", f"{sp500_metrics['hit_rate']:.1f}%",
     f"{strat_metrics['hit_rate'] - sp500_metrics['hit_rate']:+.1f}%"],
]

table = ax5.table(cellText=metrics_data, cellLoc='center', loc='center',
                  colWidths=[0.35, 0.22, 0.22, 0.21])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Header styling
for i in range(4):
    table[(0, i)].set_facecolor('#34495E')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(metrics_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ECF0F1')
        # Highlight differences
        if j == 3:  # Difference column
            val = metrics_data[i][j]
            if '+' in val:
                table[(i, j)].set_facecolor('#D5F4E6')  # Light green
            elif val.startswith('-') and i != 4:  # Better if negative (except drawdown)
                table[(i, j)].set_facecolor('#FADBD8')  # Light red

ax5.set_title('Performance Comparison Summary', fontsize=12, fontweight='bold', pad=20)

# Overall title
fig.suptitle('Hackathon Strategy vs S&P 500 Benchmark Comparison',
             fontsize=16, fontweight='bold', y=0.98)

# Save
plt.tight_layout()
plt.savefig('figs/optimized2_vs_sp500.png', dpi=220, bbox_inches='tight')
print(f"✓ Saved: figs/optimized2_vs_sp500.png")

# Print summary
print("\n" + "=" * 70)
print("STRATEGY VS S&P 500 COMPARISON")
print("=" * 70)
print(f"\n{'Metric':<20} {'Strategy':>15} {'S&P 500':>15} {'Difference':>15}")
print("-" * 70)
for i in range(1, len(metrics_data)):
    print(f"{metrics_data[i][0]:<20} {metrics_data[i][1]:>15} {metrics_data[i][2]:>15} {metrics_data[i][3]:>15}")
print("=" * 70)

# Alpha and Beta
merged['excess_strat'] = merged['port_ls'] - merged['rf'].fillna(0)
merged['excess_mkt'] = merged['ret'] - merged['rf'].fillna(0)

# Simple regression for beta
from numpy.linalg import lstsq
X = merged[['excess_mkt']].values
y = merged['excess_strat'].values
mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
X_clean = X[mask]
y_clean = y[mask]
X_with_const = np.column_stack([np.ones(len(X_clean)), X_clean])
coeffs, _, _, _ = lstsq(X_with_const, y_clean, rcond=None)
alpha_monthly = coeffs[0]
beta = coeffs[1]

print(f"\nMarket Beta: {beta:.3f}")
print(f"Monthly Alpha: {alpha_monthly * 100:.3f}% ({alpha_monthly * 12 * 100:.2f}% annualized)")
print("=" * 70)

plt.close()
