import pandas as pd
import glob
import os
import re
import numpy as np
from tqdm import tqdm

# --- Configuration ---

# --- !! ACTION REQUIRED !! ---
# 1. Path to your CANADIAN embedding batches (the folder from Part 1)
EMBEDDING_BATCH_DIR = r'C:\_Files\School\Competitions\FIAM2025\global_embeds' # <-- UPDATE THIS

# 2. Path to your North American mapping file
MAP_PATH = r'C:\_Files\School\Competitions\FIAM2025\data\mapping\North America Company Name Merge by DataDate-GVKEY-IID.csv'

# 3. Path to your quantitative (returns) data
RET_SAMPLE_PATH = r'C:\_Files\School\Competitions\FIAM2025\data\quant_data\ret_sample.parquet'

# 4. Path to save the final, merged file
OUTPUT_FILE_PATH = r'C:\_Files\School\Competitions\FIAM2025\final_merged_data_canada.pkl'

# --- !! NEW: DEBUG MODE !! ---
# Set this to a country you are processing
DEBUG_COUNTRY_NAME = 'Canada' 
# --- End Debug Mode ---


# --- End Configuration ---


def normalize_name(name):
    """
    Cleans and standardizes company names for fuzzy matching.
    e.g. "Volvo Car AB - Ordinary Shares - Class B" -> "VOLVO CAR"
    e.g. "LUZERNER KANTONALBANK" -> "LUZERNER KANTONALBANK"
    """
    if not isinstance(name, str):
        return None

    name = name.upper()
    
    # 1. Take only the part *before* the first hyphen
    # This is key to removing " - Ordinary Shares - Class B"
    name = name.split(' - ')[0]
    
    # 2. Remove all punctuation (.,')
    name = re.sub(r'[^\w\s]', '', name)
    
    # 3. Remove common *legal* suffixes. We do this *after* splitting
    # on the hyphen to avoid issues.
    suffixes = [
        'AB', 'ASA', 'PLC', 'LLC', 'INC', 'LTD', 'AG', 'NV', 'SA', 'SPA', 'SE', 'BV'
    ]
    pattern = r'\b(' + r'|'.join(suffixes) + r')\b'
    name = re.sub(pattern, '', name)
    
    # 4. Clean up extra whitespace
    name = ' '.join(name.split())
    
    return name

def load_all_batches(batch_directory):
    """
    Loads all embedding .pkl files from a directory into one DataFrame.
    """
    print(f"Loading all embedding batches from '{batch_directory}'...")
    # This will search recursively, so it works for the nested text data folder
    all_batch_files = glob.glob(os.path.join(batch_directory, '**', '*.pkl'), recursive=True)
    if not all_batch_files:
        print(f"ERROR: No '*.pkl' files found in '{batch_directory}'.")
        print("Please update the 'EMBEDDING_BATCH_DIR' variable.")
        return pd.DataFrame()
        
    df_list = []
    for f in tqdm(all_batch_files, desc="Loading batch files"):
        try:
            df_list.append(pd.read_pickle(f))
        except Exception as e:
            print(f"Warning: Could not load {f}. Error: {e}")
            
    if not df_list:
        print("ERROR: No files were successfully loaded.")
        return pd.DataFrame()
        
    df_embeddings = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df_embeddings)} total rows from {len(all_batch_files)} batch files.")
    return df_embeddings

def load_na_map(file_path):
    """
    Loads and prepares the North American mapping CSV.
    This file links names to gvkey and iid.
    """
    print(f"Loading NA map from '{file_path}'...")
    try:
        # This file *has* a header, as shown in the screenshot
        df_map = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Error loading CSV with header: {e}")
        return pd.DataFrame()
    
    # Keep only the *unique combinations* of name, gvkey, and iid.
    id_cols = ['conm', 'gvkey', 'iid']
 
    if not all(col in df_map.columns for col in id_cols):
        print(f"ERROR: NA map is missing required columns (conm, gvkey, iid). Found: {df_map.columns}")
        return pd.DataFrame()
        
    df_map_unique = df_map[id_cols].drop_duplicates().copy()
    print(f"NA map contains {len(df_map_unique)} unique securities.")
    return df_map_unique

def load_ret_sample(file_path):
    """
    Loads and prepares the final returns Parquet file.
    """
    print(f"Loading returns sample from '{file_path}'...")
    df_ret = pd.read_parquet(file_path)
    
    # Convert the linking date 'char_eom' to datetime.
    if 'char_eom' not in df_ret.columns:
        print("CRITICAL ERROR: 'char_eom' column not found in ret_sample.parquet!")
        return pd.DataFrame()
        
    df_ret['char_eom'] = pd.to_datetime(df_ret['char_eom'], format='%Y%m%d', errors='coerce')
    print(f"Loaded {len(df_ret)} return observations.")
    
    return df_ret

def main():
    df_embeddings = load_all_batches(EMBEDDING_BATCH_DIR)
    if df_embeddings.empty: return

    df_na_map = load_na_map(MAP_PATH)
    if df_na_map.empty: return

    df_ret_sample = load_ret_sample(RET_SAMPLE_PATH)
    if df_ret_sample.empty: return

    # --- Pre-processing for Merge ---
    
    print("\n--- Preparing Data for Merging ---")
    
    # A. Prepare Embeddings DataFrame
    if 'country' not in df_embeddings.columns:
        print("ERROR: 'country' column not found in loaded embedding files.")
        return
    if 'company_name' not in df_embeddings.columns:
        print("ERROR: 'company_name' column not found in loaded embedding files.")
        return
    if 'period_end_date' not in df_embeddings.columns:
        print("ERROR: 'period_end_date' column not found in loaded embedding files.")
        return

    if pd.api.types.is_string_dtype(df_embeddings['country']):
        df_embeddings['country'] = df_embeddings['country'].str.strip()

    # We just need to normalize the name
    df_embeddings['normalized_name'] = df_embeddings['company_name'].apply(normalize_name)
    df_embeddings['period_end_date'] = pd.to_datetime(df_embeddings['period_end_date'])
    
    # B. Prepare NA Map DataFrame
    df_na_map['normalized_name'] = df_na_map['conm'].apply(normalize_name)
    
    # Drop rows that failed normalization
    df_embeddings = df_embeddings.dropna(subset=['normalized_name', 'period_end_date'])
    df_na_map = df_na_map.dropna(subset=['normalized_name', 'gvkey', 'iid'])

    print(f"Embeddings ready: {len(df_embeddings)} valid rows.")
    print(f"NA map ready: {len(df_na_map)} valid rows.")
    
    
    # --- !! START DEBUG BLOCK 1 (Names) !! ---
    print("\n--- DEBUGGING MERGE 1 ---")
    
    # 2. Side-by-side name comparison
    if DEBUG_COUNTRY_NAME:
        print(f"\n--- DEBUG: Comparing names for {DEBUG_COUNTRY_NAME} ---")
        
        print(f"\n--- {DEBUG_COUNTRY_NAME} Names from EMBEDDINGS ---")
        # Filter embeddings to only the country we are debugging
        emb_sample = df_embeddings[df_embeddings['country'] == DEBUG_COUNTRY_NAME][['normalized_name', 'company_name']].drop_duplicates().head(15)
        print(emb_sample)
        
        print(f"\n--- {DEBUG_COUNTRY_NAME}-relevant Names from NA MAP ---")
        # Find names in the map that *match* the sample from our embeddings
        map_sample = df_na_map[df_na_map['normalized_name'].isin(emb_sample['normalized_name'])][['normalized_name', 'conm']].drop_duplicates().head(15)
        print(map_sample)
        if map_sample.empty:
            print("DEBUG: No name matches found in NA Map for the first 15 embedding names.")
            
    print("--- END DEBUG BLOCK 1 ---\n")
    
    # --- !! END DEBUG BLOCK 1 !! ---


    # --- Merge 1: Embeddings + NA Map ---
    print("\n--- MERGE 1: Linking Embeddings to GVKEY/IID ---")
    
    # We merge *only* on 'normalized_name' because this map file does not have 'fic'
    df_merged_1 = pd.merge(
        df_embeddings,
        df_na_map[['normalized_name', 'gvkey', 'iid']],
        on=['normalized_name'],
        how='left' # Keep all embeddings, even if they don't match
    )
    
    # Report on merge success
    matched_rows = df_merged_1['gvkey'].notna().sum()
    print(f"MERGE 1 STATS: Successfully linked {matched_rows} / {len(df_merged_1)} embedding rows to a gvkey/iid.")
    
    # Filter to only the successful matches
    df_linked_filings = df_merged_1.dropna(subset=['gvkey', 'iid'])
    print(f"Proceeding with {len(df_linked_filings)} linked filings.")
    
    
    # --- !! START NEW DEBUG BLOCK 2 (Dates) !! ---
    print("\n--- DEBUGGING MERGE 2 (DATES) ---")
    
    if df_linked_filings.empty:
        print("DEBUG: Skipping Date check, Merge 1 returned 0 rows.")
    else:
        # Get the first matched (gvkey, iid) pair
        first_match = df_linked_filings.iloc[0]
        test_gvkey = first_match['gvkey']
        test_iid = first_match['iid']
        
        print(f"DEBUG: Checking dates for a matched security: gvkey={test_gvkey}, iid={test_iid}")
        
        # 1. Get dates from our FILINGS for this security
        filing_dates = df_linked_filings[
            (df_linked_filings['gvkey'] == test_gvkey) &
            (df_linked_filings['iid'] == test_iid)
        ]['period_end_date'].dt.date.unique()
        filing_dates.sort()
        
        print(f"\n--- Dates from FILINGS (period_end_date) ---")
        print(filing_dates)
        
        # 2. Get dates from the RETURN data for this *same* security
        return_dates = df_ret_sample[
            (df_ret_sample['gvkey'] == test_gvkey) &
            (df_ret_sample['iid'] == test_iid)
        ]['char_eom'].dt.date.unique()
        return_dates.sort()
        
        print(f"\n--- Dates from RET_SAMPLE (char_eom) ---")
        print(return_dates)
        
        # 3. Check for any overlap
        overlap = set(filing_dates).intersection(set(return_dates))
        if not overlap:
            print("\nDEBUG: CRITICAL: No date overlap found for this security!")
            print("This is the reason Merge 2 is failing.")
        else:
            print(f"\nDEBUG: SUCCESS: Found {len(overlap)} overlapping dates!")
            print(f"Example overlap: {list(overlap)[0]}")

    print("--- END DEBUG BLOCK 2 ---\n")
    

    # --- Merge 2: Linked Filings + Returns ---
    print("\n--- MERGE 2: Linking Filings to Returns ---")
    
    if df_linked_filings.empty:
        print("Skipping Merge 2: No linked filings from Merge 1.")
    else:
        df_final_data = pd.merge(
            df_linked_filings,
            df_ret_sample,
            left_on=['gvkey', 'iid', 'period_end_date'],
            right_on=['gvkey', 'iid', 'char_eom'],
            how='inner' # We ONLY want rows that exist in both datasets
        )
        print(f"Total final rows linked to a T+1 return: {len(df_final_data)}")
        
        if not df_final_data.empty:
            # Save the final dataset
            df_final_data.to_pickle(OUTPUT_FILE_PATH)
            print(f"\nSuccessfully saved final merged data to:")
            print(f"{OUTPUT_FILE_PATH}")
        else:
            print("WARNING: Final dataset is empty (Merge 2 failed).")

    # --- Final Report ---
    print("\n--- MERGING COMPLETE ---")
    if 'df_final_data' not in locals() or df_final_data.empty:
         print("Total final rows linked to a T+1 return: 0")
         print("WARNING: Final dataset is empty.")


if __name__ == "__main__":
    main()