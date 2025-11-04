# This script trains a model ONLY on text embeddings to generate a downside risk score.
# It uses a custom target variable as a proxy for the Sortino ratio objective.

import pandas as pd
import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import glob


if __name__ == "__main__":
    quant_data_dir = r'C:\_Files\School\Competitions\FIAM2025\data\quant_data'
    mapping_data_dir = r'C:\_Files\School\Competitions\FIAM2025\data\mapping'
    ret_sample_path = os.path.join(quant_data_dir, 'ret_sample.csv')
    cik_link_path = os.path.join(mapping_data_dir, 'cik_gvkey_linktable_USA_only.csv')
    embeddings_batch_dir = r'C:\_Files\School\Competitions\FIAM2025\embeddings\embedding_batches_us-fin-roberta_ADVANCED_BATCHED' # Currently set to use finBERT embeds

    # Load All Pre-processed Data 
    print("--- Loading Quantitative Data ---")
    
    required_quant_cols = ['id', 'date', 'gvkey', 'year', 'month', 'stock_ret']
    df_quant = pd.read_csv(ret_sample_path, usecols=required_quant_cols)
    df_quant['date_dt'] = pd.to_datetime(df_quant['date'], format='%Y%m%d')

    print(f"--- Loading Pre-computed Text Embeddings from '{embeddings_batch_dir}' ---")
    batch_files = glob.glob(os.path.join(embeddings_batch_dir, 'batch_*.pkl'))
    if not batch_files:
        print(f"Error: No embedding batch files found in '{embeddings_batch_dir}'")
        exit()
    
    df_embeddings = pd.concat([pd.read_pickle(f) for f in batch_files])
    print(f"Loaded {len(df_embeddings)} total embeddings.")

    print("--- Loading CIK-GVKEY Link Table ---")
    df_link = pd.read_csv(cik_link_path)
    df_link = df_link.drop_duplicates(subset=['cik'])

    # Merge Datasets to Create the Master DataFrame
    print("\n--- Creating Master DataFrame ---")
    df_embeddings['year_month'] = pd.to_datetime(df_embeddings['date'], format='%Y%m%d').dt.to_period('M')
    df_quant['year_month'] = df_quant['date_dt'].dt.to_period('M')
    
    df_embeddings['cik'] = pd.to_numeric(df_embeddings['cik'], errors='coerce')
    df_link['cik'] = pd.to_numeric(df_link['cik'], errors='coerce')
    df_text_linked = pd.merge(df_embeddings, df_link, on='cik', how='inner')
    
    df_master = pd.merge(df_quant, df_text_linked.drop(columns=['cik', 'date']), on=['gvkey', 'year_month'], how='inner')
    
    embedding_cols = [f'rf_embedding_{i}' for i in range(768)]
    
    print(f"Master DataFrame created with shape: {df_master.shape}")
    print("NOTE: The DataFrame now only contains data for US stocks with available text filings.")

    # Create the Downside Risk Target Variable
    ret_var = "stock_ret"
    df_master['downside_risk_target'] = np.where(df_master[ret_var] < 0, df_master[ret_var]**2, 0)
    
    # Train Model using Expanding Window
    print("\n--- Starting Expanding Window Training ---")
    
    feature_columns = embedding_cols
    target_var = "downside_risk_target"

    starting = pd.to_datetime("20050101", format="%Y%m%d")
    counter = 0
    pred_out = pd.DataFrame()

    while (starting + pd.DateOffset(years=13 + counter)) <= pd.to_datetime("20260101", format="%Y%m%d"):
        cutoff_start_train = starting
        cutoff_end_train = starting + pd.DateOffset(years=10 + counter)
        cutoff_end_validate = starting + pd.DateOffset(years=12 + counter)
        cutoff_end_test = starting + pd.DateOffset(years=13 + counter)
        
        print(f"\nProcessing window: Training until {cutoff_end_train.year}, Testing year {cutoff_end_test.year-1}")

        train = df_master[(df_master["date_dt"] >= cutoff_start_train) & (df_master["date_dt"] < cutoff_end_train)].dropna(subset=[target_var])
        validate = df_master[(df_master["date_dt"] >= cutoff_end_train) & (df_master["date_dt"] < cutoff_end_validate)].dropna(subset=[target_var])
        test = df_master[(df_master["date_dt"] >= cutoff_end_validate) & (df_master["date_dt"] < cutoff_end_test)]

        if test.empty:
            print("  > Test set is empty. Ending training loop.")
            break

        X_train = train[feature_columns]
        Y_train = train[target_var]
        X_val = validate[feature_columns]
        Y_val = validate[target_var]
        X_test = test[feature_columns]
        
        reg = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            early_stopping_rounds=20,
            n_jobs=-1,
            tree_method="hist", 
            device="cuda"
        )
        
        print("  > Training XGBoost model to learn downside risk from text...")
        reg.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=False)
        
        risk_scores = reg.predict(X_test)
        
        test_scores = test[['id', 'date_dt', ret_var, target_var]].copy()
        test_scores['risk_score'] = risk_scores
        pred_out = pd.concat([pred_out, test_scores])
        
        counter += 1

    print("\n--- Training complete. Saving generated risk scores. ---")
    pred_out.rename(columns={'date_dt': 'date'}, inplace=True)
    pred_out['date'] = pred_out['date'].dt.strftime('%Y%m%d').astype(int)
    
    pred_out.to_csv("text_based_risk_scores.csv", index=False)
    print("Risk scores saved to 'text_based_risk_scores.csv'")

    # Evaluate the score's ability to predict downside risk
    oos_results = pred_out.dropna(subset=[target_var])
    y_true = oos_results[target_var]
    y_pred = oos_results['risk_score']
    
    oos_r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum(y_true**2))
    
    print(f"\nOut-of-Sample R-squared (predicting downside risk): {oos_r2:.4%}")


