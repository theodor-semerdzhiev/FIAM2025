import pandas as pd
import os
import glob
import pickle
import numpy as np 
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
            with open(file_path, 'rb') as f:
                df = pd.read_pickle(f)
                df_list.append(df)
        except Exception as e:
            print(f"    > Error loading {os.path.basename(file_path)}: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)

def get_text_embedding(text, model, tokenizer):
    """
    Converts a block of text into a 768-dimension numerical vector.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.zeros(768, dtype=np.float32) 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Model's max length is 512 tokens
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state[0][0].cpu().numpy()
    return embedding


if __name__ == "__main__":
    # Define file paths
    data_dir = r'C:\_Files\Personal\Projects\FIAM\FIAM2025\data'
    text_data_path = os.path.join(data_dir, 'text_data')
    output_embeddings_path = 'risk_factor_embeddings.pkl'

    print("\n--- Loading Text Data (this may take a moment) ---")
    df_text = load_all_text_data(text_data_path)
    if df_text.empty:
        print("Text data loading failed. Exiting.")
        exit()
    print(f"Loaded a total of {len(df_text)} filings.")
    

    df_text_valid = df_text[df_text['rf'].notna() & (df_text['rf'].str.len() > 100)].copy()
    print(f"Found {len(df_text_valid)} filings with valid 'Risk Factors' text to process.")

    print("\n--- Preparing for Text-to-Vector Conversion ---")
    print("Loading FinBERT model... (This may take a moment on the first run)")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")
    print("Model loaded.")
    if torch.cuda.is_available():
        print("GPU is available. Using CUDA for faster processing.")
    else:
        print("GPU not found. Processing will be done on the CPU (this may be slow).")

    print(f"\n--- Generating {len(df_text_valid)} Embeddings ---")
    embeddings = df_text_valid['rf'].progress_apply(lambda text: get_text_embedding(text, model, tokenizer))
    
    print("\n--- Creating Final Embeddings DataFrame ---")
    embedding_df = pd.DataFrame(embeddings.tolist(), index=df_text_valid.index)
    embedding_df.columns = [f'rf_embedding_{i}' for i in range(embedding_df.shape[1])]
    
    df_final_embeddings = pd.concat([df_text_valid[['cik', 'date']], embedding_df], axis=1)
    df_final_embeddings.to_pickle(output_embeddings_path)
    
    print(f"\nEmbeddings DataFrame saved to '{output_embeddings_path}'")

    print("\n--- Example of the final DataFrame with embedding features ---")
    print(df_final_embeddings[['cik', 'date', 'rf_embedding_0', 'rf_embedding_1', 'rf_embedding_2']].head())
    
    print("\nFeature engineering complete.")
    print("This embeddings file can now be loaded and merged with the quantitative data for model training.")

