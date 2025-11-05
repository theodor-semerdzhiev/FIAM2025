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
MODEL_PATH = r'D:\market_data\text_data\CHECKPOINTS\GERMANIC_MAINLAND-fin-roberta\checkpoint-3786'

# --- !! 2. SET THE COUNTRIES FOR THIS MODEL !! ---
# COUNTRIES_TO_PROCESS = ['China', 'Hong Kong', 'Japan', 'Singapore', 'South Korea', 'Taiwan'] # <-- SET THIS!
# COUNTRIES_TO_PROCESS = ['Canada']
# COUNTRIES_TO_PROCESS = ['United Kingdom', 'Ireland', 'Australia']
COUNTRIES_TO_PROCESS = ['Germany', 'Austria', 'Switzerland', 'Netherlands', 'Belgium', 'Luxembourg']

# --- !! 3. SET YOUR DATA AND OUTPUT PATHS !! ---
TEXT_DATA_PATH = r'C:\_Files\School\Competitions\FIAM2025\data\global_text_data' 

# --- !! 4. SET THE TEXT COLUMN TO EMBED !! ---
TEXT_COLUMN_NAME = 'section_7_risk_factors' 

# --- !! 5. BATCHING CONFIGURATION !! ---
PANDAS_BATCH_SIZE = 5000 # How many ROWS to process before saving a file
GPU_BATCH_SIZE = 360      # How many CHUNKS to feed to the GPU at once.
                         # Increase this to max out your VRAM (e.g., 128, 256)

# --- !! 6. CHUNK SIZE (in words) !! ---
# We split the text into chunks of this size. 400 words is safely under the 512 token limit.
CHUNK_SIZE_IN_WORDS = 400

# The script will auto-name this based on the countries
BATCH_DIR = f"embedding_batches_{'_'.join(COUNTRIES_TO_PROCESS).lower()}_ADVANCED_BATCHED"

# --- End Configuration ---


def load_all_text_data_filtered(data_directory, countries_list):
    """
    Loads all pickle files, *immediately* filters them for the specified countries,
    and combines only those rows into a single pandas DataFrame.
    This is much more memory-efficient than loading everything first.
    """
    all_pickle_files = sorted(glob.glob(os.path.join(data_directory, '**', '*.pkl'), recursive=True))
    
    if not all_pickle_files:
        print(f"No pickle files found in '{data_directory}'.")
        return pd.DataFrame()

    df_list = []
    print(f"Found {len(all_pickle_files)} pickle files. Loading and filtering...")
    for i, file_path in enumerate(all_pickle_files):
        try:
            df = pd.read_pickle(file_path)
            
            # Check if required 'country' column exists
            if 'country' not in df.columns:
                print(f"  > Skipping {os.path.basename(file_path)} (missing 'country' column)")
                continue
                
            # Filter for the specified countries
            filtered_df = df[df['country'].isin(countries_list)]
            
            if not filtered_df.empty:
                df_list.append(filtered_df)
                
        except Exception as e:
            print(f"    > Error loading or filtering {os.path.basename(file_path)}: {e}")
            
    if not df_list:
        print(f"No data found for countries: {', '.join(countries_list)}")
        return pd.DataFrame()
        
    print(f"Successfully loaded {len(df_list)} filtered DataFrames.")
    return pd.concat(df_list, ignore_index=True)


if __name__ == "__main__":
    
    if MODEL_PATH == r'PLEASE_SET_YOUR_LATEST_CHECKPOINT_PATH':
        print(f"ERROR: Please open '{__file__}' and set the 'MODEL_PATH' variable at the top.")
        exit()
    if TEXT_DATA_PATH == r'C:\_Files\School\Competitions\FIAM2025\data\international_text_data':
        print(f"ERROR: Please open '{__file__}' and set the 'TEXT_DATA_PATH' variable at the top.")
        exit()
    if not COUNTRIES_TO_PROCESS:
        print(f"ERROR: Please open '{__file__}' and set the 'COUNTRIES_TO_PROCESS' list at the top.")
        exit()


    print("\n--- Loading And Filtering Text Data ---")
    df_text = load_all_text_data_filtered(TEXT_DATA_PATH, COUNTRIES_TO_PROCESS)
    if df_text.empty: exit()
    
    # --- !! NEW FILTERING LOGIC !! ---
    print(f"Total rows for {', '.join(COUNTRIES_TO_PROCESS)}: {len(df_text)}")
    
    # 1. Check if TEXT_COLUMN_NAME exists
    if TEXT_COLUMN_NAME not in df_text.columns:
        print(f"CRITICAL ERROR: The '{TEXT_COLUMN_NAME}' column was not found in your DataFrame.")
        print("Please check the 'TEXT_COLUMN_NAME' variable. Aborting.")
        exit()
        
    # 2. Filter for valid text
    df_text_valid = df_text[
        df_text[TEXT_COLUMN_NAME].notna() & (df_text[TEXT_COLUMN_NAME].str.len() > 100)
    ].copy()
    
    print(f"Found {len(df_text_valid)} filings with valid '{TEXT_COLUMN_NAME}' text to process.")
    
    if df_text_valid.empty:
        print("No valid text found for these countries. Aborting.")
        exit()
    # --- !! END NEW FILTERING LOGIC !! ---


    print("\n--- Initializing Custom Model ---")
    print(f"Loading model from: {MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        model.to(device)
        model.eval() # Set model to evaluation mode
    else:
        print("WARNING: GPU not found. Processing will be on CPU (slow).")

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
                chunk_batch = all_chunks[k : k + GPU_BATCH_SIZE]
                
                inputs = tokenizer(chunk_batch, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
                
                outputs = model(**inputs)
                
                # Get the [CLS] token embedding for *all items in the batch*
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
        
        embedding_df = pd.DataFrame(final_pooled_embeddings, index=batch_df.index)
        embedding_df.columns = [f'rf_embedding_{j}' for j in range(embedding_df.shape[1])]
        
        # Keep the columns you identified as necessary for the merge
        columns_to_keep = ['company_name', 'period_end_date', 'country']
        
        # Check if all required columns exist in this batch
        missing_cols = [col for col in columns_to_keep if col not in batch_df.columns]
        if missing_cols:
            print(f"WARNING: Batch is missing required metadata columns: {missing_cols}")
            # Keep whatever columns *are* available from your list
            columns_to_keep = [col for col in columns_to_keep if col in batch_df.columns]

        # Only select the columns that are confirmed to exist
        if columns_to_keep:
            batch_info_df = batch_df[columns_to_keep].copy()
            
            # --- Date Normalization ---
            # Ensure period_end_date is a proper datetime object for future merging
            if 'period_end_date' in batch_info_df.columns:
                batch_info_df['period_end_date'] = pd.to_datetime(batch_info_df['period_end_date'], errors='coerce')
            # --- End Date Normalization ---

            batch_result_df = pd.concat([batch_info_df, embedding_df], axis=1)
        else:
            print("WARNING: No metadata columns found. Saving only embeddings.")
            batch_result_df = embedding_df
        # --- !! END MODIFICATION !! ---
        
        batch_result_df.to_pickle(batch_file_path)
        print(f"Pandas Batch {i+1} saved to {batch_file_path}")

    print("\n--- All batches processed successfully! ---")
    print(f"The output is a folder named '{BATCH_DIR}' containing all the processed embedding files.")


