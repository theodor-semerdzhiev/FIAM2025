# --- imports ---
from __future__ import annotations
import os
import re
from typing import Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------
# Config (edit as needed)
# ----------------------------
BASE_DIR = "./data"
FEATURE  = "mgmt"

LINKTABLE_CSV   = f"{BASE_DIR}/cik_gvkey_linktable_USA_only.csv"
EMBEDDINGS_FMT  = f"{BASE_DIR}/embeddings/{{year}}_{FEATURE}_embeddings.pkl"   # e.g., 2005_mgmt_embeddings.pkl
QUANT_PARQUET   = f"{BASE_DIR}/ret_sample.parquet"                              # big panel
OUT_DIR         = "./data/training_data_surprise_model"                         # where we save outputs
os.makedirs(OUT_DIR, exist_ok=True)

# Parallelism
MAX_PROCS = 2   # <-- cap active processes here

# ----------------------------
# Helpers
# ----------------------------
def pack_embeddings(df: pd.DataFrame, feature: str = "mgmt",
                    out_col: str | None = None, dtype=np.float32
                   ) -> Tuple[pd.DataFrame, np.ndarray, list[str]]:
    prefix = f"{feature}_embedding_"
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")

    idx_cols = []
    for c in df.columns:
        m = pattern.match(c)
        if m:
            idx_cols.append((int(m.group(1)), c))
    if not idx_cols:
        raise ValueError(f"No embedding columns found with prefix '{prefix}'")

    idx_cols.sort(key=lambda t: t[0])
    emb_cols = [c for _, c in idx_cols]

    X = df.loc[:, emb_cols].to_numpy(dtype=dtype, copy=False)
    arr_col = out_col or f"{feature}_embedding"

    df_packed = df.drop(columns=emb_cols).copy()
    df_packed[arr_col] = list(X)
    return df_packed, X, emb_cols


def prepare_linktable(df_link: pd.DataFrame) -> pd.DataFrame:
    df = df_link.copy()
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["date"] = df["datadate"].dt.to_period("M").astype(str)
    df.drop(columns=["datadate"], inplace=True)
    df["cik"] = df["cik"].astype("Int64")
    return df


def prepare_embeddings(emb_path: str, feature: str = "mgmt") -> pd.DataFrame:
    emb = pd.read_pickle(emb_path)
    emb["date"] = pd.to_datetime(emb["date"], format="%Y%m%d")
    emb["date"] = emb["date"].dt.to_period("M").astype(str)
    emb["cik"]  = emb["cik"].astype("Int64")
    packed, _, _ = pack_embeddings(emb, feature=feature)
    return packed


def prepare_quant_panel(quant_path: str, year: int) -> pd.DataFrame:
    dfq = pd.read_parquet(quant_path, engine="fastparquet")
    dfq = dfq[dfq["date"].dt.year == year].copy()
    dfq["date"] = dfq["date"].dt.to_period("M").astype(str)
    return dfq


def build_training_dfs(year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    linktable = pd.read_csv(LINKTABLE_CSV)
    linktable = prepare_linktable(linktable)

    emb_path = EMBEDDINGS_FMT.format(year=year)
    packed_df = prepare_embeddings(emb_path, feature=FEATURE)

    dfq = prepare_quant_panel(QUANT_PARQUET, year)

    joined = packed_df.merge(
        linktable,
        on=["cik", "date"],
        how="left",
        suffixes=("", "_linktable"),
    )
    joined = joined.merge(
        dfq[["gvkey", "date", "ni_be"]],
        on=["gvkey", "date"],
        how="left",
    )
    joined = joined.dropna(subset=["ni_be"])

    final = joined[["cik", "gvkey", "date", "file_type", f"{FEATURE}_embedding", "ni_be"]]
    training_10K = final[final["file_type"] == "10K"].drop(columns="file_type").reset_index(drop=True)
    training_10K = training_10K.drop_duplicates(subset=["cik", "gvkey", "date", "ni_be"])
    training_10Q = final[final["file_type"] == "10Q"].drop(columns="file_type").reset_index(drop=True)
    training_10Q = training_10Q.drop_duplicates(subset=["cik", "gvkey", "date", "ni_be"])
    return training_10K, training_10Q


def save_training_dfs(year: int, df_10k: pd.DataFrame, df_10q: pd.DataFrame) -> tuple[int, int]:
    out_10k = os.path.join(OUT_DIR, f"{year}_{FEATURE}_training_10K.parquet")
    out_10q = os.path.join(OUT_DIR, f"{year}_{FEATURE}_training_10Q.parquet")
    df_10k.to_parquet(out_10k, index=False)
    df_10q.to_parquet(out_10q, index=False)
    return len(df_10k), len(df_10q)


def process_year(year: int) -> dict:
    """Worker entry: build + save; return a small summary dict (or error)."""
    try:
        t10k, t10q = build_training_dfs(year)
        n10k, n10q = save_training_dfs(year, t10k, t10q)
        return {"year": year, "n10k": n10k, "n10q": n10q, "status": "ok", "error": None}
    except Exception as e:
        return {"year": year, "n10k": 0, "n10q": 0, "status": "error", "error": repr(e)}


# ----------------------------
# Run for many years with a cap on active processes
# ----------------------------
if __name__ == "__main__":
    YEARS = list(range(2005, 2026))  # adjust as needed

    results = []
    with ProcessPoolExecutor(max_workers=MAX_PROCS) as ex:
        futs = {ex.submit(process_year, y): y for y in YEARS}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing years"):
            results.append(fut.result())

    # Optional: print a compact summary
    results.sort(key=lambda d: d["year"])
    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] != "ok"]

    print("\n--- Summary ---")
    for r in ok:
        print(f"{r['year']}: 10K={r['n10k']:>6}, 10Q={r['n10q']:>6}")
    if err:
        print("\nErrors:")
        for r in err:
            print(f"{r['year']}: {r['error']}")
