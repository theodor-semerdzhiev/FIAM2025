import pandas as pd
import os
import glob
import pickle
import numpy as np
import shutil
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

tqdm.pandas()

def load_all_text_data(data_directory):
    """
    Loads all pickle files containing text data from a directory and combines
    them into a single pandas DataFrame.
    """
    all_pickle_files = sorted(glob.glob(os.path.join(data_directory, '**', '*.pkl'), recursive=True))
    
    if not all_pickle_files:
        print(f"No pickle files found in '{data_directory}'.")
        return pd.DataFrame()

    df_list = []
    print(f"Found {len(all_pickle_files)} pickle files. Loading...")
    for i, file_path in enumerate(all_pickle_files):
        print(f"  > Loading {os.path.basename(file_path)} ({i+1}/{len(all_pickle_files)})...")
        try:
            df = pd.read_pickle(file_path)
            df_list.append(df)
        except Exception as e:
            print(f"    > Error loading {os.path.basename(file_path)}: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)

def get_text_embedding(text, model, tokenizer):
    """
    Converts a block of text into a 768-dimension numerical vector (embedding).
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.zeros(768, dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state[0][0].cpu().numpy()
    return embedding

if __name__ == "__main__":
    data_dir = r'C:\_Files\Personal\Projects\FIAM\FIAM2025\data'
    text_data_path = os.path.join(data_dir, 'text_data')
    
    BATCH_SIZE = 1000
    BATCH_DIR = 'embedding_batches'

    print("\n--- Loading Text Data ---")
    df_text = load_all_text_data(text_data_path)
    if df_text.empty: exit()
    
    df_text_valid = df_text[df_text['rf'].notna() & (df_text['rf'].str.len() > 100)].copy()
    print(f"Found {len(df_text_valid)} filings with valid 'Risk Factors' text to process.")

    print("\n--- Initializing FinBERT Model ---")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: GPU not found. Processing will be on CPU (slow).")

    print(f"\n--- Generating {len(df_text_valid)} Embeddings in batches of {BATCH_SIZE} ---")
    os.makedirs(BATCH_DIR, exist_ok=True)
    num_batches = int(np.ceil(len(df_text_valid) / BATCH_SIZE))

    for i in range(num_batches):
        batch_start = i * BATCH_SIZE
        batch_end = (i + 1) * BATCH_SIZE
        batch_df = df_text_valid.iloc[batch_start:batch_end]
        
        batch_file_path = os.path.join(BATCH_DIR, f'batch_{i+1}.pkl')
        
        if os.path.exists(batch_file_path):
            print(f"Batch {i+1}/{num_batches} already processed. Skipping.")
            continue

        print(f"Processing Batch {i+1}/{num_batches}...")
        
        embeddings = batch_df['rf'].progress_apply(lambda text: get_text_embedding(text, model, tokenizer))
        
        embedding_df = pd.DataFrame(embeddings.tolist(), index=batch_df.index)
        embedding_df.columns = [f'rf_embedding_{j}' for j in range(embedding_df.shape[1])]
        
        batch_result_df = pd.concat([batch_df[['cik', 'date']], embedding_df], axis=1)
        
        batch_result_df.to_pickle(batch_file_path)
        print(f"Batch {i+1} saved to {batch_file_path}")

    print("\n--- All batches processed successfully! ---")
    print(f"The output is a folder named '{BATCH_DIR}' containing all the processed embedding files.")

