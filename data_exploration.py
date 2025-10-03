import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = "/Users/loading.../Documents/Projects/FIAM2025/data/ret_sample.csv"
chunksize = 1_000_000

# --- Step 1: Count rows ---
print("Counting rows...")
row_count = sum(1 for _ in open(file)) - 1  # minus header
print(f"Total rows: {row_count:,}")

# --- Step 2: Inspect first rows for structure ---
sample_head = pd.read_csv(file, nrows=1000)
print("\n--- Column Info ---")
print(sample_head.info())
print(sample_head.head())

# --- Step 2b: Count unique stock IDs ---
print("\n--- Counting unique stock IDs (gvkey + iid) ---")
unique_ids = set()

for chunk in pd.read_csv(file, chunksize=chunksize, usecols=["gvkey", "iid"]):
    # Create tuple pairs of (gvkey, iid)
    pairs = list(zip(chunk["gvkey"], chunk["iid"]))
    unique_ids.update(pairs)

print(f"Number of unique stock IDs: {len(unique_ids):,}")



# --- Step 3: Descriptive stats (streaming) ---
print("\n--- Descriptive Stats (numeric cols) ---")
stats = {}
for chunk in pd.read_csv(file, chunksize=chunksize):
    for col in chunk.columns:
        if pd.api.types.is_numeric_dtype(chunk[col]):
            col_stats = stats.setdefault(col, {"count":0, "sum":0, "sum2":0, "min":np.inf, "max":-np.inf})
            vals = chunk[col].dropna().values
            if len(vals) > 0:
                col_stats["count"] += len(vals)
                col_stats["sum"] += vals.sum()
                col_stats["sum2"] += (vals**2).sum()
                col_stats["min"] = min(col_stats["min"], vals.min())
                col_stats["max"] = max(col_stats["max"], vals.max())

for col, s in stats.items():
    mean = s["sum"]/s["count"]
    var = s["sum2"]/s["count"] - mean**2
    std = np.sqrt(var)
    print(f"{col}: count={s['count']}, mean={mean:.3f}, std={std:.3f}, "
          f"min={s['min']:.3f}, max={s['max']:.3f}")

# --- Step 4: Sample for quick plots ---
print("\nSampling ~1% of data for visualizations...")
sample = pd.read_csv(file, skiprows=lambda i: i>0 and np.random.rand() > 0.01)

# Distribution of stock returns
sample['stock_ret'].hist(bins=100)
plt.title("Distribution of Stock Returns")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.show()

# Average stock return per year
if "year" in sample.columns:
    sample.groupby("year")["stock_ret"].mean().plot(kind="line", marker="o")
    plt.title("Average Stock Return by Year")
    plt.xlabel("Year")
    plt.ylabel("Mean Return")
    plt.show()
