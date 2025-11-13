#!/usr/bin/env python3
"""
Generate Monthly Holdings CSV for Hackathon Submission
Format: year, month, stock_id, weight, position
"""

import pandas as pd

print("Generating monthly holdings CSV for hackathon submission...")

# Load changes data
changes_df = pd.read_csv("csv/changes_HACKATHON.csv")

# Load monthly data to get portfolio sizes
monthly_df = pd.read_csv("csv/monthly_ls_HACKATHON.csv")

# Filter for stocks that were "added" or "kept" (i.e., in the portfolio that month)
holdings_df = changes_df[changes_df["change"].isin(["added", "kept"])].copy()

print(f"  Total position-months: {len(holdings_df)}")

# Merge with monthly data to get portfolio sizes
holdings_df = holdings_df.merge(
    monthly_df[["year", "month", "n_long", "n_short"]],
    on=["year", "month"],
    how="left"
)

# Calculate equal weights within each side
# Long side: 150% / n_long stocks
# Short side: 50% / n_short stocks
LONG_WEIGHT = 1.50
SHORT_WEIGHT = 0.50

holdings_df["weight"] = holdings_df.apply(
    lambda r: LONG_WEIGHT / r["n_long"] if r["side"] == "long" else -SHORT_WEIGHT / r["n_short"],
    axis=1
)

# Select and rename columns for submission
submission_df = holdings_df[["year", "month", "id", "weight", "side"]].copy()
submission_df = submission_df.rename(columns={"id": "stock_id", "side": "position"})

# Sort by year, month, position, weight (descending)
submission_df = submission_df.sort_values(["year", "month", "position", "weight"], ascending=[True, True, True, False])

# Save
output_file = "csv/monthly_holdings_submission.csv"
submission_df.to_csv(output_file, index=False)

print(f"  ✓ Saved: {output_file}")
print(f"  Rows: {len(submission_df):,}")
print(f"  Unique months: {submission_df.groupby(['year', 'month']).ngroups}")
print(f"  Unique stocks: {submission_df['stock_id'].nunique():,}")

# Verification: check weights sum correctly for a few months
print("\n  Verification (sample months):")
for (year, month), group in submission_df.groupby(["year", "month"]).head(3):
    long_sum = group[group["position"] == "long"]["weight"].sum()
    short_sum = group[group["position"] == "short"]["weight"].sum()
    net_exposure = long_sum + short_sum  # shorts are negative
    print(f"    {year}-{month:02d}: Long={long_sum:+.2%}, Short={short_sum:+.2%}, Net={net_exposure:+.2%}")

# Show sample
print("\n  Sample holdings (2025-06):")
sample = submission_df[(submission_df["year"] == 2025) & (submission_df["month"] == 6)].head(10)
print(sample.to_string(index=False))

print("\n✅ Monthly holdings CSV ready for hackathon submission!")
