import os
import glob
import pickle
import threading
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

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
OUT_DIR = "./data/embeddings"
os.makedirs(OUT_DIR, exist_ok=True)

# how many years to process in parallel
MAX_PARALLEL_YEARS = 5  # <-- tweak this

# ----------------------------
# Helpers
# ----------------------------
def chunked(seq: List[str], n: int) -> Iterable[List[str]]:
    """Yield successive n-sized chunks from seq."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def load_all_text_data(data_directory: str, year: str | None = None) -> pd.DataFrame:
    """Load & concat all pickle files for a year (or all)."""
    pattern = (os.path.join(data_directory, '**', '*.pkl')
               if year is None else
               os.path.join(data_directory, '**', f'*{year}.pkl'))
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print(f"[MAIN] No pickle files found for year={year} in '{data_directory}'.")
        return pd.DataFrame()

    dfs = []
    print(f"[MAIN] Found {len(files)} files for year={year}. Loading...")
    for fp in files:
        try:
            with open(fp, "rb") as f:
                dfs.append(pd.read_pickle(f))
        except Exception as e:
            print(f"[MAIN]   Error loading {os.path.basename(fp)}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

# ----------------------------
# Embedding utils (shared)
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
MODEL = AutoModel.from_pretrained("ProsusAI/finbert").to(DEVICE)
MODEL.eval()

def get_text_embedding(text: str) -> np.ndarray:
    """Return a 768-dim embedding for a single text (CLS token)."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.zeros(768, dtype=np.float32)

    with torch.no_grad():
        inputs = TOKENIZER(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        ).to(DEVICE)
        outputs = MODEL(**inputs)
        emb = outputs.last_hidden_state[0][0].detach().cpu().numpy().astype(np.float32)
        return emb

# ----------------------------
# Worker (thread target)
# ----------------------------
def process_year(year: str, df_year_copy: pd.DataFrame, tqdm_position: int = 0):
    """Compute embeddings for one year's precomputed DataFrame copy."""
    try:
        if df_year_copy.empty:
            print(f"[{year}] No valid rows to process. Skipping.")
            return

        total = len(df_year_copy)
        print(f"[{year}] Generating embeddings for {total} rows on device={DEVICE}...")

        embeddings = [
            get_text_embedding(txt)
            for txt in tqdm(
                df_year_copy[FEATURE],
                desc=f"Year {year}",
                position=tqdm_position,
                leave=True
            )
        ]

        emb_mat = np.stack(embeddings, axis=0) if embeddings else np.empty((0, 768), dtype=np.float32)
        emb_cols = [f"{FEATURE}_embedding_{i}" for i in range(emb_mat.shape[1] if emb_mat.size else 768)]
        embedding_df = pd.DataFrame(emb_mat, index=df_year_copy.index[:len(embeddings)], columns=emb_cols)

        df_final = pd.concat([df_year_copy.loc[embedding_df.index, EXTRA_FEATURES], embedding_df], axis=1)

        out_path = os.path.join(OUT_DIR, f"{year}_{FEATURE}_embeddings.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(df_final, f)
        print(f"[{year}] Saved embeddings -> {out_path}")

        preview_cols = EXTRA_FEATURES + [f"{FEATURE}_embedding_{i}" for i in range(min(3, len(emb_cols)))]
        print(f"[{year}] Sample:\n{df_final[preview_cols].head()}")

    except Exception as e:
        print(f"[{year}] ERROR: {e}")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("[MAIN] Preparing DataFrames (load + filter) before starting threads...")
    print(f"[MAIN] Using device: {DEVICE}")

    # Precompute per-year filtered DataFrames in MAIN thread
    year_to_df: Dict[str, pd.DataFrame] = {}
    for yr in YEARS:
        print(f"\n[MAIN] --- Year {yr}: load & prefilter ---")
        df = load_all_text_data(TEXT_DIR, year=yr)
        if df.empty:
            print(f"[MAIN] Year {yr}: no data.")
            year_to_df[yr] = pd.DataFrame()
            continue

        print(f"[MAIN] Year {yr}: loaded {len(df)} filings.")
        if FEATURE not in df.columns:
            print(f"[MAIN] Year {yr}: column '{FEATURE}' missing. Skipping year.")
            year_to_df[yr] = pd.DataFrame()
            continue

        df_valid = df[df[FEATURE].notna() & (df[FEATURE].astype(str).str.len() > 100)].copy(deep=True)

        for col in EXTRA_FEATURES:
            if col not in df_valid.columns:
                df_valid[col] = None

        print(f"[MAIN] Year {yr}: {len(df_valid)} rows with valid '{FEATURE}' text.")
        year_to_df[yr] = df_valid

    print("\n[MAIN] Processing in batches...")
    batch_idx = 0
    for batch_years in chunked(YEARS, MAX_PARALLEL_YEARS):
        print(f"\n[MAIN] === Batch {batch_idx + 1}: {batch_years} ===")
        threads: List[threading.Thread] = []

        # Create & start threads for this batch; distinct tqdm positions 0..len(batch)-1
        for pos_in_batch, yr in enumerate(batch_years):
            df_valid = year_to_df.get(yr, pd.DataFrame())
            df_copy = df_valid.copy(deep=True)  # hand a copy to the thread
            t = threading.Thread(target=process_year, args=(yr, df_copy, pos_in_batch))
            threads.append(t)
            t.start()

        # Wait for the whole batch to finish before starting the next
        for t in threads:
            t.join()

        batch_idx += 1

    print("\n[MAIN] All batches complete. Embeddings ready.")
