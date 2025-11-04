import pandas as pd
import os
import glob
import pickle
import numpy as np
import shutil
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import math

tqdm.pandas()

# --- Configuration ---

# --- !! 1. SET YOUR MODEL PATH !! ---
MODEL_PATH = r'D:\market_data\text_data\CHECKPOINTS\USA-fin-roberta-filings-2016\checkpoint-3020000'

# --- !! 2. SET THE COUNTRIES FOR THIS MODEL !! ---
COUNTRIES_TO_PROCESS = ['United States'] # <-- SET THIS!

# --- !! 3. SET YOUR DATA AND OUTPUT PATHS !! ---
TEXT_DATA_PATH = r'C:\_Files\Personal\Projects\FIAM\FIAM2025\data\all_pickle_data_combined' # <-- SET THIS!

# --- !! 4. SET THE TEXT COLUMN TO EMBED !! ---
TEXT_COLUMN_NAME = 'section_7_risk_factors' 

# --- !! 5. BATCHING CONFIGURATION !! ---
PANDAS_BATCH_SIZE = 5000 # How many ROWS to process before saving a file
GPU_BATCH_SIZE = 64      # How many CHUNKS to feed to the GPU at once.
                         # Increase this to max out your VRAM (e.g., 128, 256)

# --- !! 6. CHUNK SIZE (in words) !! ---
# We split the text into chunks of this size. 400 words is safely under the 512 token limit.
CHUNK_SIZE_IN_WORDS = 400

# The script will auto-name this based on the countries
BATCH_DIR = f"embedding_batches_{'_'.join(COUNTRIES_TO_PROCESS).lower()}_ADVANCED_BATCHED"

# --- End Configuration ---


def load_all_text_data(data_directory, countries_to_load, required_text_col):
    """
    Loads all pickle files from a directory, filters them in memory for
    the specified countries, and combines them into a single pandas DataFrame.
    """
    all_pickle_files = sorted(glob.glob(os.path.join(data_directory, '**', '*.pkl'), recursive=True))
    
    if not all_pickle_files:
        print(f"No pickle files found in '{data_directory}'.")
        return pd.DataFrame()

    df_list = []
    print(f"Found {len(all_pickle_files)} pickle files. Scanning for countries: {', '.join(countries_to_load)}")
    
    total_rows_found = 0
    
    for i, file_path in enumerate(all_pickle_files):
        try:
            df = pd.read_pickle(file_path)
            
            if 'country' not in df.columns or required_text_col not in df.columns:
                print(f"  > Skipping {os.path.basename(file_path)} (missing 'country' or '{required_text_col}' column)")
                continue

            filtered_df = df[df['country'].isin(countries_to_load)]
            
            if not filtered_df.empty:
                print(f"  > Found {len(filtered_df)} matching rows in {os.path.basename(file_path)} ({i+1}/{len(all_pickle_files)})")
                df_list.append(filtered_df)
                total_rows_found += len(filtered_df)
            
        except Exception as e:
            print(f"    > Error loading or filtering {os.path.basename(file_path)}: {e}")
            
    if not df_list:
        print("No matching data found in any file.")
        return pd.DataFrame()
        
    print(f"\n--- Load complete. Found a total of {total_rows_found:,} matching rows. ---")
    return pd.concat(df_list, ignore_index=True)

# --- (get_pooled_embedding function is removed as logic is now in the main loop) ---

if __name__ == "__main__":
    
    if MODEL_PATH == r'PLEASE_SET_YOUR_LATEST_CHECKPOINT_PATH':
        print(f"ERROR: Please open '{__file__}' and set the 'MODEL_PATH' variable at the top.")
        exit()
    if TEXT_DATA_PATH == r'C:\_Files\Personal\Projects\FIAM\FIAM2025\data\all_pickle_data_combined':
        print(f"ERROR: Please open '{__file__}' and set the 'TEXT_DATA_PATH' variable at the top.")
        exit()
    if not COUNTRIES_TO_PROCESS:
        print(f"ERROR: Please open '{__file__}' and set the 'COUNTRIES_TO_PROCESS' list at the top.")
        exit()

    # --- Load Data ---
    print("\n--- Loading Text Data ---")
    df_text = load_all_text_data(TEXT_DATA_PATH, COUNTRIES_TO_PROCESS, TEXT_COLUMN_NAME)
    if df_text.empty: exit()
    
    print(f"Total rows for {', '.join(COUNTRIES_TO_PROCESS)}: {len(df_text)}")
    
    df_text_valid = df_text[
        df_text[TEXT_COLUMN_NAME].notna() & (df_text[TEXT_COLUMN_NAME].str.len() > 100)
    ].copy()
    
    print(f"Found {len(df_text_valid)} filings with valid '{TEXT_COLUMN_NAME}' text to process.")
    
    if df_text_valid.empty:
        print(f"No valid '{TEXT_COLUMN_NAME}' text found for these countries. Aborting.")
        exit()

    # --- Load Model ---
    print("\n--- Initializing Custom Model ---")
    print(f"Loading model from: {MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        model.to(device)
        model.eval() # Set model to evaluation mode (disables dropout, etc.)
    else:
        print("WARNING: GPU not found. Processing will be on CPU (slow).")

    # --- Process Batches ---
    print(f"\n--- Generating {len(df_text_valid)} POOLED Embeddings ---")
    print(f"Processing in pandas chunks of {PANDAS_BATCH_SIZE}")
    print(f"Processing in GPU batches of {GPU_BATCH_SIZE}")
    os.makedirs(BATCH_DIR, exist_ok=True)
    num_batches = int(np.ceil(len(df_text_valid) / PANDAS_BATCH_SIZE))

    for i in range(num_batches):
        batch_start = i * PANDAS_BATCH_SIZE
        batch_end = (i + 1) * PANDAS_BATCH_SIZE
        batch_df = df_text_valid.iloc[batch_start:batch_end]
        
        batch_file_path = os.path.join(BATCH_DIR, f'batch_{i+1}.pkl')
        
        if os.path.exists(batch_file_path):
            print(f"Pandas Batch {i+1}/{num_batches} already processed. Skipping.")
            continue

        print(f"\n--- Processing Pandas Batch {i+1}/{num_batches} ---")
        
        # --- !! NEW BATCHING LOGIC !! ---
        
        # 1. Get all texts from the pandas batch
        all_texts = batch_df[TEXT_COLUMN_NAME].tolist()
        
        # 2. Create one giant list of all chunks
        all_chunks = []
        # Keep track of which chunks belong to which filing
        filing_to_chunk_indices = []
        
        print(" 1. Chunking texts...")
        for text in all_texts:
            if not isinstance(text, str) or len(text.strip()) == 0:
                filing_to_chunk_indices.append((len(all_chunks), len(all_chunks))) # Mark as empty
                continue
                
            words = text.split()
            total_chunks_for_this_filing = math.ceil(len(words) / CHUNK_SIZE_IN_WORDS)
            
            chunk_start_index = len(all_chunks)
            for j in range(total_chunks_for_this_filing):
                chunk_words = words[j * CHUNK_SIZE_IN_WORDS : (j + 1) * CHUNK_SIZE_IN_WORDS]
                all_chunks.append(" ".join(chunk_words))
            filing_to_chunk_indices.append((chunk_start_index, len(all_chunks)))

        print(f" 2. Created {len(all_chunks)} total chunks from {len(all_texts)} filings.")
        
        # 3. Process all chunks in GPU batches
        print(f" 3. Running {len(all_chunks)} chunks through GPU in batches of {GPU_BATCH_SIZE}...")
        all_chunk_embeddings = []
        with torch.no_grad():
            for k in tqdm(range(0, len(all_chunks), GPU_BATCH_SIZE), desc="GPU Batches"):
                # Grab a batch of chunk strings
                chunk_batch = all_chunks[k : k + GPU_BATCH_SIZE]
                
                # Tokenize the whole batch at once
                inputs = tokenizer(chunk_batch, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
                
                # Get embeddings for the whole batch
                outputs = model(**inputs)
                
                # Get the [CLS] token embedding for *all items in the batch*
                # Shape: [GPU_BATCH_SIZE, 768]
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                all_chunk_embeddings.extend(cls_embeddings)
        
        # 4. Pool the embeddings back together
        print(" 4. Pooling embeddings by filing...")
        all_chunk_embeddings_np = np.array(all_chunk_embeddings)
        final_pooled_embeddings = []
        
        for start_idx, end_idx in filing_to_chunk_indices:
            if start_idx == end_idx:
                # This was an empty or invalid text
                pooled_embedding = np.zeros(768, dtype=np.float32)
            else:
                # Get all embeddings for this filing and average them
                chunk_embeddings_for_this_filing = all_chunk_embeddings_np[start_idx:end_idx]
                pooled_embedding = np.mean(chunk_embeddings_for_this_filing, axis=0)
            
            final_pooled_embeddings.append(pooled_embedding)
        
        # --- !! END NEW BATCHING LOGIC !! ---
        
        # Convert list of embeddings into a new DataFrame
        embedding_df = pd.DataFrame(final_pooled_embeddings, index=batch_df.index)
        embedding_df.columns = [f'{TEXT_COLUMN_NAME}_embedding_{j}' for j in range(embedding_df.shape[1])]
        
        columns_to_keep = ['country']
            
        batch_info_df = batch_df[columns_to_keep]
        batch_result_df = pd.concat([batch_info_df, embedding_df], axis=1)
        
        batch_result_df.to_pickle(batch_file_path)
        print(f"Pandas Batch {i+1} saved to {batch_file_path}")

    print("\n--- All batches processed successfully! ---")
    print(f"The output is a folder named '{BATCH_DIR}' containing all the processed embedding files.")

