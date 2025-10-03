#!/usr/bin/env python3
"""
Generate Vector Similarity Scores for Stock Returns
Simple script that reads ret_sample.csv and generates similarity scores
Output: CSV with id, date, and various similarity scores for easy merging
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import os
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

def load_ret_sample() -> pd.DataFrame:
    """Load the return sample data"""
    print("Loading ret_sample.csv...")
    
    # Try different possible file names
    possible_files = [
        'ret_sample.csv',
        'data/ret_sample.csv', 
        'ret_sample-002.parquet',
        'data/ret_sample-002.parquet'
    ]
    
    df = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            print(f"Found data file: {file_path}")
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            break
    
    if df is None:
        raise FileNotFoundError(f"Could not find data file. Tried: {possible_files}")
    
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded data: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique securities: {df['id'].nunique()}")
    
    return df

class SimpleSimilarityScorer:
    """
    Simplified similarity scoring system focused on generating scores for merging
    """
    
    def __init__(self, lookback_months=12, similarity_threshold=0.75):
        self.lookback_months = lookback_months
        self.similarity_threshold = similarity_threshold
        self.scaler = StandardScaler()
        
    def calculate_similarity_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate similarity scores for all (id, date) combinations
        """
        print("Calculating similarity scores...")
        
        # Get factor columns (exclude id, date, stock_ret, and other non-factor columns)
        exclude_cols = {'id', 'date', 'stock_ret', 'ret_eom', 'year', 'month', 'gvkey', 'iid', 'permno', 'ticker', 'cusip', 'excntry'}
        factor_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        print(f"Using {len(factor_cols)} factor columns for similarity calculation")
        print(f"Factor columns: {factor_cols[:10]}..." if len(factor_cols) > 10 else f"Factor columns: {factor_cols}")
        
        # Sort by id and date
        df = df.sort_values(['id', 'date']).copy()
        
        # Initialize result columns
        result_data = []
        
        # Process each security individually
        unique_ids = df['id'].unique()
        
        for i, security_id in enumerate(unique_ids):
            if i % 100 == 0:
                print(f"Processing security {i+1}/{len(unique_ids)}: {security_id}")
            
            # Get security data
            sec_data = df[df['id'] == security_id].copy().reset_index(drop=True)
            
            if len(sec_data) < 12:  # Need minimum history
                # Add zeros for securities with insufficient data
                for idx in range(len(sec_data)):
                    result_data.append({
                        'id': security_id,
                        'date': sec_data.iloc[idx]['date'],
                        'similarity_score': 0.0,
                        'similarity_confidence': 0.0,
                        'similarity_matches': 0,
                        'similarity_volatility': 0.0,
                        'similarity_strength': 0.0
                    })
                continue
            
            # Fill missing factor values with cross-sectional median
            for col in factor_cols:
                if col in sec_data.columns:
                    sec_data[col] = sec_data[col].fillna(sec_data[col].median())
            
            # Get feature matrix
            feature_matrix = sec_data[factor_cols].values
            
            # Skip if all NaN
            if np.isnan(feature_matrix).all():
                for idx in range(len(sec_data)):
                    result_data.append({
                        'id': security_id,
                        'date': sec_data.iloc[idx]['date'],
                        'similarity_score': 0.0,
                        'similarity_confidence': 0.0,
                        'similarity_matches': 0,
                        'similarity_volatility': 0.0,
                        'similarity_strength': 0.0
                    })
                continue
            
            # Fit scaler on security's data
            try:
                feature_matrix_clean = np.nan_to_num(feature_matrix, 0)
                scaler = StandardScaler().fit(feature_matrix_clean)
                feature_matrix_scaled = scaler.transform(feature_matrix_clean)
            except:
                # Fallback for scaling issues
                for idx in range(len(sec_data)):
                    result_data.append({
                        'id': security_id,
                        'date': sec_data.iloc[idx]['date'],
                        'similarity_score': 0.0,
                        'similarity_confidence': 0.0,
                        'similarity_matches': 0,
                        'similarity_volatility': 0.0,
                        'similarity_strength': 0.0
                    })
                continue
            
            # Calculate similarity scores for each date
            for idx in range(len(sec_data)):
                current_date = sec_data.iloc[idx]['date']
                current_features = feature_matrix_scaled[idx]
                current_return = sec_data.iloc[idx]['stock_ret'] if 'stock_ret' in sec_data.columns else 0
                
                # Get lookback window
                lookback_start = current_date - pd.DateOffset(months=self.lookback_months)
                lookback_mask = (sec_data['date'] >= lookback_start) & (sec_data['date'] < current_date)
                lookback_indices = sec_data.index[lookback_mask].tolist()
                
                if len(lookback_indices) < 3:  # Need minimum history
                    similarity_score = 0.0
                    confidence = 0.0
                    matches = 0
                    volatility = 0.0
                    strength = 0.0
                else:
                    # Get historical features and returns
                    hist_features = feature_matrix_scaled[lookback_indices]
                    hist_returns = sec_data.iloc[lookback_indices]['stock_ret'].values if 'stock_ret' in sec_data.columns else np.zeros(len(lookback_indices))
                    
                    # Calculate similarities
                    similarities = cosine_similarity([current_features], hist_features)[0]
                    
                    # Find similar periods
                    similar_mask = similarities >= self.similarity_threshold
                    
                    if not np.any(similar_mask):
                        # Use top 3 most similar if no exact matches
                        top_k = min(3, len(similarities))
                        top_indices = np.argsort(similarities)[-top_k:]
                        similar_returns = hist_returns[top_indices]
                        similar_sims = similarities[top_indices]
                    else:
                        similar_returns = hist_returns[similar_mask]
                        similar_sims = similarities[similar_mask]
                    
                    # Calculate weighted prediction
                    if len(similar_returns) > 0 and similar_sims.sum() > 0:
                        weights = similar_sims / similar_sims.sum()
                        similarity_score = np.average(similar_returns, weights=weights)
                        confidence = len(similar_returns) / len(lookback_indices)
                        matches = len(similar_returns)
                        volatility = np.std(similar_returns) if len(similar_returns) > 1 else 0.0
                        strength = abs(similarity_score) * confidence
                    else:
                        similarity_score = 0.0
                        confidence = 0.0
                        matches = 0
                        volatility = 0.0
                        strength = 0.0
                
                # Store results
                result_data.append({
                    'id': security_id,
                    'date': current_date,
                    'similarity_score': similarity_score,
                    'similarity_confidence': confidence,
                    'similarity_matches': matches,
                    'similarity_volatility': volatility,
                    'similarity_strength': strength
                })
        
        # Convert to DataFrame
        result_df = pd.DataFrame(result_data)
        
        print(f"Generated similarity scores for {len(result_df):,} observations")
        print(f"Non-zero similarity scores: {(result_df['similarity_score'] != 0).sum():,} ({(result_df['similarity_score'] != 0).mean():.1%})")
        
        return result_df

def add_cross_sectional_scores(df: pd.DataFrame, similarity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-sectional similarity features based on the individual scores
    """
    print("Adding cross-sectional similarity features...")
    
    # Merge similarity scores with original data
    merged = df.merge(similarity_df, on=['id', 'date'], how='left')
    
    # Add cross-sectional rankings and relative scores
    for date in merged['date'].unique():
        date_mask = merged['date'] == date
        date_data = merged[date_mask]
        
        if len(date_data) > 10:  # Need minimum observations for ranking
            # Similarity score rank percentile
            merged.loc[date_mask, 'similarity_rank_pct'] = date_data['similarity_score'].rank(pct=True)
            
            # Similarity score relative to cross-sectional median
            median_score = date_data['similarity_score'].median()
            merged.loc[date_mask, 'similarity_vs_median'] = date_data['similarity_score'] - median_score
            
            # High confidence similarity indicator
            high_conf_threshold = date_data['similarity_confidence'].quantile(0.8)
            merged.loc[date_mask, 'high_confidence_similarity'] = (date_data['similarity_confidence'] >= high_conf_threshold).astype(float)
    
    # Fill NaNs
    similarity_features = ['similarity_rank_pct', 'similarity_vs_median', 'high_confidence_similarity']
    for col in similarity_features:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
    
    # Create final output with just the similarity columns
    output_cols = ['id', 'date'] + [col for col in merged.columns if 'similarity' in col.lower()]
    result_df = merged[output_cols].copy()
    
    print(f"Added {len(similarity_features)} cross-sectional features")
    return result_df

def main():
    """Main function to generate similarity scores CSV"""
    print("="*60)
    print("VECTOR SIMILARITY SCORE GENERATOR")
    print("="*60)
    
    # Load data
    df = load_ret_sample()
    
    # Initialize similarity scorer
    scorer = SimpleSimilarityScorer(lookback_months=12, similarity_threshold=0.75)
    
    # Calculate similarity scores
    similarity_df = scorer.calculate_similarity_scores(df)
    
    # Add cross-sectional features
    final_df = add_cross_sectional_scores(df, similarity_df)
    
    # Save to CSV
    output_file = 'similarity_scores.csv'
    final_df.to_csv(output_file, index=False)
    
    print(f"\n" + "="*60)
    print(f"SIMILARITY SCORES GENERATED")
    print(f"="*60)
    print(f"Output file: {output_file}")
    print(f"Shape: {final_df.shape}")
    print(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
    print(f"Securities: {final_df['id'].nunique()}")
    
    # Summary statistics
    print(f"\nSimilarity Score Statistics:")
    sim_stats = final_df['similarity_score'].describe()
    print(f"Mean: {sim_stats['mean']:.4f}")
    print(f"Std:  {sim_stats['std']:.4f}")
    print(f"Min:  {sim_stats['min']:.4f}")
    print(f"Max:  {sim_stats['max']:.4f}")
    print(f"Non-zero: {(final_df['similarity_score'] != 0).sum():,} ({(final_df['similarity_score'] != 0).mean():.1%})")
    
    print(f"\nColumns in output:")
    for col in final_df.columns:
        non_zero_pct = (final_df[col] != 0).mean() if final_df[col].dtype in ['float64', 'int64'] else 1.0
        print(f"  {col}: {non_zero_pct:.1%} non-zero")
    
    print(f"\nReady to merge with your existing data using:")
    print(f"  similarity_scores = pd.read_csv('{output_file}')")
    print(f"  merged_data = your_data.merge(similarity_scores, on=['id', 'date'], how='left')")

if __name__ == "__main__":
    main()
