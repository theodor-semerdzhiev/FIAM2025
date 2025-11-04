import pandas as pd
import glob
import os
import re
import numpy as np
from tqdm import tqdm

# --- Configuration ---

# 1. Path to your embedding batches (the folder from Part 1)
EMBEDDING_BATCH_DIR = r'C:\_Files\School\Competitions\FIAM2025\global_embeds'

# 2. Path to your Global mapping file
GLOBAL_MAP_PATH = r'C:\_Files\School\Competitions\FIAM2025\data\mapping\Global (ex Canada and US) Company Name Merge by DataDate-GVKEY-IID.csv'

# 3. Path to your quantitative (returns) data
RET_SAMPLE_PATH = r'C:\_Files\School\Competitions\FIAM2025\data\quant_data\ret_sample.parquet'

# 4. Path to save the final, merged file
OUTPUT_FILE_PATH = r'C:\_Files\School\Competitions\FIAM2025\final_merged_data.pkl'

# --- !! NEW: DEBUG MODE !! ---
# Set this to a country you are processing, e.g., 'China'
DEBUG_COUNTRY_NAME = 'China' 
# --- End Debug Mode ---


# --- End Configuration ---


# This map is built from Appendix A of the hackathon PDF.
# It's crucial for merging 'country' (from filings) with 'fic' (from global map)
COUNTRY_TO_FIC_MAP = {
    'Austria': 'AUT',
    'Australia': 'AUS',
    'Belgium': 'BEL',
    'China': 'CHN',
    'Canada': 'CAN',
    'Denmark': 'DNK',
    'Finland': 'FIN',
    'France': 'FRA',
    'Germany': 'DEU',
    'Hong Kong': 'HKG',
    'Ireland': 'IRL',
    'Italy': 'ITA',
    'Israel': 'ISL',
    'Japan': 'JPN',
    'South Korea': 'KOR',
    'United Kingdom': 'GBR',
    'Luxembourg': 'LUX',
    'Mexico': 'MEX',
    'Netherlands': 'NLD',
    'New Zealand': 'NZL',
    'Norway': 'NOR',
    'Portugal': 'PRT',
    'Spain': 'ESP',
    'Singapore': 'SGP',
    'Sweden': 'SWE',
    'Taiwan': 'TWN',
    'Switzerland': 'CHE',
    # Add any other mappings if needed
}

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

def load_global_map(file_path):
    """
    Loads and prepares the Global mapping CSV.
    This file links names to gvkey and iid.
    """
    print(f"Loading global map from '{file_path}'...")
    try:
        # Assuming the CSV has headers as seen in your screenshot:
        # fic,gvkey,datadate,iid,conm
        # UPDATE: Your screenshot shows NO header. The `except` block is correct.
        df_map = pd.read_csv(file_path)
    except Exception as e:
        # This block should be running
        print(f"Info: Error loading CSV with header (expected): {e}")
        print("Attempting to load without a header...")
        df_map = pd.read_csv(file_path, header=None)
        # Manually assign headers based on your screenshots
        # The first column is empty in your screenshot, let's call it 'fic'
        # and assume the CSV reader handles the leading comma.
        # If the first col is BLANK, use this:
        # df_map.columns = ['blank', 'fic', 'gvkey', 'datadate', 'iid', 'conm'] + [f'col_{i}' for i in range(6, len(df_map.columns))]
        # Let's try the original assumption:
        try:
             df_map.columns = ['fic', 'gvkey', 'datadate', 'iid', 'conm'] + [f'col_{i}' for i in range(5, len(df_map.columns))]
        except:
             print("CRITICAL ERROR: Column count mismatch loading global_map. Check file.")
             print(f"File has {len(df_map.columns)} columns.")
             return pd.DataFrame()

    
    # Keep only the *unique combinations* of name, fic, gvkey, and iid.
    # This slims down the file from monthly rows to one row per *security*.
    # This is the key to solving your share class problem.
    id_cols = ['conm', 'fic', 'gvkey', 'iid']
    if not all(col in df_map.columns for col in id_cols):
        print(f"ERROR: Global map is missing required columns. Found: {df_map.columns}")
        return pd.DataFrame()
        
    # --- !! POTENTIAL BUG FIX: Strip whitespace from FIC !! ---
    # Sometimes "CHN " != "CHN"
    if pd.api.types.is_string_dtype(df_map['fic']):
        df_map['fic'] = df_map['fic'].str.strip()
    # --- End Fix ---
        
    df_map_unique = df_map[id_cols].drop_duplicates().copy()
    print(f"Global map contains {len(df_map_unique)} unique securities.")
    return df_map_unique

def load_ret_sample(file_path):
    """
    Loads and prepares the final returns Parquet file.
    """
    print(f"Loading returns sample from '{file_path}'...")
    df_ret = pd.read_parquet(file_path)
    
    # Convert the linking date 'char_eom' to datetime.
    # This is the 'time t' date, which matches our 'period_end_date'.
    df_ret['char_eom'] = pd.to_datetime(df_ret['char_eom'])
    print(f"Loaded {len(df_ret)} return observations.")
    return df_ret

def main():
    # 1. Load Embeddings
    df_embeddings = load_all_batches(EMBEDDING_BATCH_DIR)
    if df_embeddings.empty: return
    
    # 2. Load Global Map
    df_global_map = load_global_map(GLOBAL_MAP_PATH)
    if df_global_map.empty: return
    
    # 3. Load Returns Data
    df_ret_sample = load_ret_sample(RET_SAMPLE_PATH)
    if df_ret_sample.empty: return

    # --- Pre-processing for Merge ---
    
    print("\n--- Preparing Data for Merging ---")
    
    # A. Prepare Embeddings DataFrame
    # - Map country name ("Sweden") to FIC code ("SWE")
    # - Normalize company name for fuzzy matching
    
    # Check for columns before processing
    if 'country' not in df_embeddings.columns:
        print("ERROR: 'country' column not found in loaded embedding files.")
        return
    if 'company_name' not in df_embeddings.columns:
        print("ERROR: 'company_name' column not found in loaded embedding files.")
        return
    if 'period_end_date' not in df_embeddings.columns:
        print("ERROR: 'period_end_date' column not found in loaded embedding files.")
        return

    # --- !! POTENTIAL BUG FIX: Strip whitespace from country !! ---
    if pd.api.types.is_string_dtype(df_embeddings['country']):
        df_embeddings['country'] = df_embeddings['country'].str.strip()
    # --- End Fix ---

    df_embeddings['fic'] = df_embeddings['country'].map(COUNTRY_TO_FIC_MAP)
    df_embeddings['normalized_name'] = df_embeddings['company_name'].apply(normalize_name)
    # Ensure date is in correct format (it should be already, but good to check)
    df_embeddings['period_end_date'] = pd.to_datetime(df_embeddings['period_end_date'])
    
    # B. Prepare Global Map DataFrame
    # - Normalize 'conm' (company name) for fuzzy matching
    df_global_map['normalized_name'] = df_global_map['conm'].apply(normalize_name)
    
    # Drop rows that failed normalization
    df_embeddings = df_embeddings.dropna(subset=['normalized_name', 'period_end_date'])
    # We drop 'fic' NaNs *after* debugging
    df_global_map = df_global_map.dropna(subset=['normalized_name', 'fic', 'gvkey', 'iid'])

    print(f"Embeddings ready: {len(df_embeddings)} valid rows.")
    print(f"Global map ready: {len(df_global_map)} valid rows.")
    
    
    # --- !! START NEW DEBUG BLOCK !! ---
    print("\n--- DEBUGGING MERGE 1 ---")
    
    # 1. Check for FIC mapping failures
    nan_fics = df_embeddings['fic'].isna().sum()
    print(f"DEBUG: Embeddings with NaN FIC (bad country name): {nan_fics} / {len(df_embeddings)}")
    if nan_fics > 0:
        print(f"DEBUG: Unique countries found in embeddings: {df_embeddings['country'].unique()}")
        print(f"DEBUG: Countries in our map: {list(COUNTRY_TO_FIC_MAP.keys())}")
        if nan_fics == len(df_embeddings):
            print("CRITICAL DEBUG: All 'fic' keys are NaN. Merge 1 will 100% fail. Check country names.")
            return # Stop before the merge

    # 2. Side-by-side name comparison
    debug_fic = COUNTRY_TO_FIC_MAP.get(DEBUG_COUNTRY_NAME)
    if debug_fic:
        print(f"\n--- DEBUG: Comparing names for {DEBUG_COUNTRY_NAME} (FIC: {debug_fic}) ---")
        
        print(f"\n--- {DEBUG_COUNTRY_NAME} Names from EMBEDDINGS ---")
        emb_sample = df_embeddings[df_embeddings['fic'] == debug_fic][['normalized_name', 'company_name']].drop_duplicates().head(15)
        print(emb_sample)
        
        print(f"\n--- {DEBUG_COUNTRY_NAME} Names from GLOBAL MAP ---")
        map_sample = df_global_map[df_global_map['fic'] == debug_fic][['normalized_name', 'conm']].drop_duplicates().head(15)
        print(map_sample)
        
        if emb_sample.empty or map_sample.empty:
            print(f"DEBUG: No data found for {DEBUG_COUNTRY_NAME} in one of the files.")
            
    else:
        print(f"DEBUG: Could not find '{DEBUG_COUNTRY_NAME}' in COUNTRY_TO_FIC_MAP.")
        
    print("--- END DEBUG BLOCK ---\n")
    
    # Now, drop NaNs from fic after we've debugged
    df_embeddings = df_embeddings.dropna(subset=['fic'])
    print(f"Embeddings (post-NaN-FIC drop): {len(df_embeddings)} valid rows.")
    if df_embeddings.empty:
        print("Stopping: No embeddings left after dropping NaN FICs.")
        return
    # --- !! END NEW DEBUG BLOCK !! ---


    # --- Merge 1: Embeddings + Global Map ---
    print("\n--- MERGE 1: Linking Embeddings to GVKEY/IID ---")
    
    df_merged_1 = pd.merge(
        df_embeddings,
        df_global_map[['normalized_name', 'fic', 'gvkey', 'iid']],
        on=['normalized_name', 'fic'],
        how='left' # Keep all embeddings, even if they don't match
    )
    
    # Report on merge success
    matched_rows = df_merged_1['gvkey'].notna().sum()
    print(f"MERGE 1 STATS: Successfully linked {matched_rows} / {len(df_merged_1)} embedding rows to a gvkey/iid.")
    
    # Filter to only the successful matches
    df_linked_filings = df_merged_1.dropna(subset=['gvkey', 'iid'])
    print(f"Proceeding with {len(df_linked_filings)} linked filings.")

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
            print("\nFinal DataFrame columns:")
            print(df_final_data.columns.tolist())
        else:
            print("WARNING: Final dataset is empty (Merge 2 failed).")

    # --- Final Report ---
    print("\n--- MERGING COMPLETE ---")
    if 'df_final_data' not in locals():
         print("Total final rows linked to a T+1 return: 0")
         print("WARNING: Final dataset is empty.")
         print("Common issues:")
         print("- Date mismatch (e.g., 2023-09-30 vs 2023-09-01) <--- CHECK THIS")
         print("- Name normalization failed (check `normalize_name` function)")
         print("- No temporal overlap between filings and returns")


if __name__ == "__main__":
    main()

