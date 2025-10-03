
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.isotonic import IsotonicRegression

sns.set(context="notebook", style="whitegrid")
RNG = 42

# ----- Configuration -----
BASE_DIR = "./data/training_data_surprise_model"
OUTPUT_DIR = "./data/rolling_window_results"
MIN_TRAIN_YEARS = 3
VALIDATION_YEARS = 1
FIRST_TEST_YEAR = 2005
LAST_TEST_YEAR = 2025

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "metrics"), exist_ok=True)

# ----- Helper Functions -----
_EPS = 1e-6

def y_to_z(y):
    """Transform y to latent space using arctanh"""
    y_ = np.clip(y, -1 + _EPS, 1 - _EPS)
    return np.arctanh(y_)

def z_to_y(z):
    """Transform latent z back to y using tanh"""
    return np.clip(np.tanh(z), -1.0, 1.0)

def report_metrics(y_true, y_pred, label=""):
    """Calculate and print regression metrics"""
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    print(f"{label}RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f} | EVS={evs:.4f}")
    return dict(rmse=rmse, mae=mae, r2=r2, evs=evs)

def concat_years(start_year: int, end_year: int, base_dir: str, file_type: str = "10K"):
    """Concatenate yearly parquet files into a single DataFrame"""
    dfs = []
    for year in range(start_year, end_year + 1):
        path = os.path.join(base_dir, f"{year}_mgmt_training_{file_type}.parquet")
        if os.path.exists(path):
            year_df = pd.read_parquet(path, engine="fastparquet")
            dfs.append(year_df)
        else:
            print(f"⚠️ Skipping missing file: {path}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def get_window_years(test_year, min_train_years=8, val_years=2):
    """
    Calculate training and validation year ranges for a given test year.
    
    Rules:
    - At least min_train_years for training
    - val_years for validation
    - Training must be at least as long as validation
    - Need at least 2 years validation if 3+ years training
    """
    train_start = test_year - min_train_years - val_years
    train_end = test_year - val_years - 1
    val_start = test_year - val_years
    val_end = test_year - 1
    
    return train_start, train_end, val_start, val_end

def create_model_pipeline(base_params:dict = None):
    """Create the model pipeline with TransformedTargetRegressor"""

    if base_params is None:
        base_pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("pca", PCA(random_state=RNG)),
            ("krr", KernelRidge(kernel="rbf"))
        ])
    else:
        base_pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("pca", PCA(n_components=base_params['regressor__pca__n_components'], random_state=RNG)),
            ("krr", KernelRidge(kernel="rbf", 
                                alpha=base_params['regressor__krr__alpha'],
                                gamma=base_params['regressor__krr__gamma']))
        ])

    
    model = TransformedTargetRegressor(
        regressor=base_pipe,
        func=y_to_z,
        inverse_func=z_to_y
    )
    
    return model

def save_model(model, test_year, output_dir):
    """Save trained model to disk"""
    model_path = os.path.join(output_dir, "models", f"model_test_{test_year}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Model saved: {model_path}")
    return model_path

def save_iso_model(model, test_year, output_dir):
    """Save trained model to disk"""
    model_path = os.path.join(output_dir, "models", f"iso_model_test_{test_year}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Model saved: {model_path}")
    return model_path

def save_metrics(metrics_dict, test_year, output_dir):
    """Save metrics to CSV"""
    metrics_path = os.path.join(output_dir, "metrics", f"metrics_test_{test_year}.csv")
    pd.DataFrame([metrics_dict]).to_csv(metrics_path, index=False)
    print(f"✓ Metrics saved: {metrics_path}")
    return metrics_path

def fit_isotonic_calibration(y_val_raw, y_val) -> IsotonicRegression:

    # 2) Fit isotonic mapping on validation set
    iso = IsotonicRegression(y_min=-1, y_max=1, out_of_bounds="clip")
    iso.fit(y_val_raw, y_val)
    return iso


# ----- Main Rolling Window Pipeline -----
def run_rolling_window_pipeline():
    """Execute the rolling window training and evaluation pipeline"""
    
    print("="*80)
    print("ROLLING WINDOW PIPELINE")
    print("="*80)
    
    num_cores = os.cpu_count()
    print(f"🖥️ Total CPU cores: {num_cores}")
    print(f"🔧 Using {num_cores - 1} cores for each iteration")
    
    # Load all available data
    print("\n📂 Loading data...")
    df = concat_years(2005, 2025, BASE_DIR)
    
    if df.empty:
        print("❌ No data loaded. Check your BASE_DIR path.")
        return
    
    # Extract year from date
    df["year"] = df["date"].astype(str).str[:4].astype(int)
    print(f"✓ Loaded data from years: {df['year'].min()} to {df['year'].max()}")
    print(f"✓ Total samples: {len(df)}")
    
    # Store all results
    all_results = []
    
    # Hyperparameter search space
    param_dist = {
        "regressor__pca__n_components": [16, 32, 48, 96, 128],
        "regressor__krr__alpha": np.geomspace(1e-3, 1e1, 10),
        "regressor__krr__gamma": np.geomspace(1e-4, 1e0, 10),
    }
    
    # Iterate through test years
    for test_year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
        print("\n" + "="*80)
        print(f"🔄 Processing Test Year: {test_year}")
        print("="*80)
        
        # Calculate window years
        train_start, train_end, val_start, val_end = get_window_years(
            test_year, MIN_TRAIN_YEARS, VALIDATION_YEARS
        )
        
        print(f"📅 Training: {train_start}-{train_end} ({train_end - train_start + 1} years)")
        print(f"📅 Validation: {val_start}-{val_end} ({val_end - val_start + 1} years)")
        print(f"📅 Test: {test_year}")
        
        # Check if we have enough data
        if train_start < df["year"].min():
            print(f"⚠️ Insufficient data for test year {test_year}. Skipping...")
            continue
        
        # Split data
        train_df = df[df["year"].between(train_start, train_end)]
        val_df = df[df["year"].between(val_start, val_end)]
        test_df = df[df["year"] == test_year]
        
        if train_df.empty or val_df.empty or test_df.empty:
            print(f"⚠️ Empty dataset for test year {test_year}. Skipping...")
            continue
        
        print(f"Train samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        # Prepare features and targets
        X_train = np.vstack(train_df["mgmt_embedding"].values).astype(np.float32)
        y_train = np.clip(train_df["ni_be"].to_numpy(np.float32), -1, 1)
        
        X_val = np.vstack(val_df["mgmt_embedding"].values).astype(np.float32)
        y_val = np.clip(val_df["ni_be"].to_numpy(np.float32), -1, 1)
        
        X_test = np.vstack(test_df["mgmt_embedding"].values).astype(np.float32)
        y_test = np.clip(test_df["ni_be"].to_numpy(np.float32), -1, 1)
        
        # Create model
        model = create_model_pipeline()
        
        # Hyperparameter search with cross-validation
        print("\n🔍 Running hyperparameter search...")

        model.fit(X_train, y_train)
        best_model = model
        
        # Evaluate on all splits
        print("\n📊 Evaluating model...")
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        y_test_pred = best_model.predict(X_test)
        
        train_metrics = report_metrics(y_train, y_train_pred, "Train → ")
        val_metrics = report_metrics(y_val, y_val_pred, "Val   → ")
        test_metrics = report_metrics(y_test, y_test_pred, "Test  → ")
        
        # Isotonic regression
        iso_model = fit_isotonic_calibration(y_val_pred, y_val)

        # Save model
        model_path = save_model(best_model, test_year, OUTPUT_DIR)
        
        iso_model_path = save_iso_model(iso_model, test_year, OUTPUT_DIR)

        # Compile results
        result = {
            "test_year": test_year,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            # "best_cv_rmse": -search.best_score_,
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "train_evs": train_metrics["evs"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "val_evs": val_metrics["evs"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_r2": test_metrics["r2"],
            "test_evs": test_metrics["evs"],
            "model_path": model_path,
            "iso_model_path": iso_model_path,
            "timestamp": datetime.now().isoformat(),
            # **{f"param_{k}": v for k, v in search.best_params_.items()}
        }
        
        all_results.append(result)
        save_metrics(result, test_year, OUTPUT_DIR)
    
    # Save summary of all results
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(OUTPUT_DIR, "all_results_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print("\n" + "="*80)
        print(f"✅ Pipeline complete! Summary saved to: {summary_path}")
        print("="*80)
        
        # Display summary statistics
        print("\n📈 Summary Statistics:")
        print(summary_df[["test_year", "test_rmse", "test_mae", "test_r2"]].to_string(index=False))
    else:
        print("\n⚠️ No results generated. Check data availability.")

# ----- Execute Pipeline -----
if __name__ == "__main__":
    run_rolling_window_pipeline()