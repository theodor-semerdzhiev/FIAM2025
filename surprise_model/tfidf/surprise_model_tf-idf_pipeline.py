import os
# ---- keep CPU threads sane (set BEFORE heavy imports) ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import glob
import pickle
from typing import List, Iterable, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------
# Config
# ----------------------------
DATA_DIR = r"/Users/tsemerdz/Projects/FIAM2025/data"
TEXT_DIR = os.path.join(DATA_DIR, "text_data")

FEATURE = "mgmt"
EXTRA_FEATURES = ['cik', 'date', 'file_type']

START_YEAR = 2005
END_YEAR = 2026
YEARS: List[str] = [str(y) for y in range(START_YEAR, END_YEAR)]

OUT_DIR = "./data/tfidf"
os.makedirs(OUT_DIR, exist_ok=True)

# number of concurrent processes (years in parallel)
MAX_PARALLEL_YEARS = 2

# TF-IDF params (start with unigrams; add bigrams later)
TFIDF_KW = dict(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 1),
    min_df=5,
    max_df=0.8,
    max_features=100_000,
    dtype=np.float32,
    norm="l2",
)

# ----------------------------
# Helpers
# ----------------------------
def chunked(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def list_year_files(data_directory: str, year: str) -> List[str]:
    pattern = os.path.join(data_directory, '**', f'*{year}.pkl')
    return sorted(glob.glob(pattern, recursive=True))

def load_all_text_data_for_year(data_directory: str, year: str) -> pd.DataFrame:
    files = list_year_files(data_directory, year)
    if not files:
        return pd.DataFrame()
    dfs = []
    # per-process tqdm while loading
    for fp in tqdm(files, desc=f"[{year}] Loading pickles", leave=False, dynamic_ncols=True):
        try:
            with open(fp, "rb") as f:
                dfs.append(pd.read_pickle(f))
        except Exception as e:
            print(f"[{year}] Error loading {os.path.basename(fp)}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def fit_tfidf(corpus: List[str]) -> Tuple[sparse.csr_matrix, TfidfVectorizer]:
    vec = TfidfVectorizer(**TFIDF_KW)
    X = vec.fit_transform(corpus)   # single heavy call
    return X, vec

def save_year_artifacts(year: str,
                        X: sparse.csr_matrix,
                        feature_names: np.ndarray,
                        meta_df: pd.DataFrame):
    mat_path = os.path.join(OUT_DIR, f"{year}_{FEATURE}_tfidf.npz")
    vocab_path = os.path.join(OUT_DIR, f"{year}_{FEATURE}_vocab.pkl")
    meta_path  = os.path.join(OUT_DIR, f"{year}_{FEATURE}_meta.pkl")

    sparse.save_npz(mat_path, X)
    with open(vocab_path, "wb") as f:
        pickle.dump(feature_names, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(meta_path, "wb") as f:
        pickle.dump(meta_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[{year}] Saved TF-IDF: shape={X.shape}, nnz={X.nnz}")

# ----------------------------
# Worker (process target)
# ----------------------------
def process_one_year(year: str) -> str:
    try:
        df = load_all_text_data_for_year(TEXT_DIR, year)
        if df.empty:
            return f"[{year}] No files."

        if FEATURE not in df.columns:
            return f"[{year}] Missing column '{FEATURE}'."

        df_valid = df[df[FEATURE].notna() & (df[FEATURE].astype(str).str.len() > 100)].copy(deep=True)
        if df_valid.empty:
            return f"[{year}] No valid '{FEATURE}' rows."

        for col in EXTRA_FEATURES:
            if col not in df_valid.columns:
                df_valid[col] = None

        texts = df_valid[FEATURE].astype(str).tolist()
        print(f"[{year}] Fitting TF-IDF on {len(texts)} documents...")
        X, vec = fit_tfidf(texts)

        feature_names = vec.get_feature_names_out()
        meta_df = df_valid[EXTRA_FEATURES].reset_index(drop=True)

        save_year_artifacts(year, X, feature_names, meta_df)

        if hasattr(vec, "idf_"):
            top_k = min(10, len(feature_names))
            rare_terms = feature_names[np.argsort(vec.idf_)[-top_k:]]
            print(f"[{year}] Rare terms sample: {list(rare_terms)}")

        return f"[{year}] OK ({X.shape[0]} docs, {X.shape[1]} terms)"
    except Exception as e:
        return f"[{year}] ERROR: {e}"

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("[MAIN] TF-IDF params:", TFIDF_KW)
    print("[MAIN] OUT_DIR:", OUT_DIR)
    print("[MAIN] MAX_PARALLEL_YEARS (processes):", MAX_PARALLEL_YEARS)

    # overall progress over all years
    overall = tqdm(total=len(YEARS), desc="Years completed", dynamic_ncols=True)

    batch_idx = 0
    for batch_years in chunked(YEARS, MAX_PARALLEL_YEARS):
        batch_idx += 1
        print(f"\n[MAIN] === Batch {batch_idx}: {batch_years} ===")

        with ProcessPoolExecutor(max_workers=len(batch_years)) as ex:
            futures = {ex.submit(process_one_year, yr): yr for yr in batch_years}
            for fut in as_completed(futures):
                yr = futures[fut]
                try:
                    msg = fut.result()
                except Exception as e:
                    msg = f"[{yr}] FAILED (exception from worker): {e}"
                print(msg)
                overall.update(1)

    overall.close()
    print("\n[MAIN] All batches complete. TF-IDF features ready.")
