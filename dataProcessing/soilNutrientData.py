import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

# =========================== LOGGING CONFIGURATION =========================== #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================== DATA LOADING =========================== #
def load_depth_data(input_dir: str) -> dict:
    """Load all depth-wise CSV files into a dictionary of DataFrames."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Directory '{input_path}' not found!")
    
    depth_files = list(input_path.glob("depth_*.csv"))
    if not depth_files:
        raise FileNotFoundError(f"No CSV files found in '{input_path}'. Expected 'depth_*.csv' files.")
    
    depth_data = {}
    for file in depth_files:
        depth = file.stem.replace("depth_", "")
        df = pd.read_csv(file)
        logger.info(f"Loaded file: {file.name} | Shape: {df.shape}")
        depth_data[depth] = df
    
    return depth_data

# =========================== MISSING VALUE HANDLING =========================== #
def handle_missing_values(df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """Handle missing values: Median for small gaps, KNN for large gaps."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputer = KNNImputer(n_neighbors=5)
    
    for col in numeric_cols:
        missing_ratio = df[col].isnull().mean()
        
        if missing_ratio == 0:
            continue  # No missing values
        elif missing_ratio < threshold:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col] = imputer.fit_transform(df[[col]])
    
    return df

# =========================== OUTLIER DETECTION & REMOVAL =========================== #
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detects and replaces outliers using the IQR method."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        if col in ["latitude", "longitude"]:
            continue  # Skip geospatial columns
        
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        df.loc[outliers, col] = df[col].median()
    
    return df

# =========================== FEATURE SCALING =========================== #
def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scales numerical features using MinMaxScaler, excluding lat/lon."""
    scaler = MinMaxScaler()
    feature_columns = df.columns.difference(["latitude", "longitude"])
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df

# =========================== PREPROCESSING PIPELINE =========================== #
def preprocess_depth_data(input_dir: str, output_dir: str) -> None:
    """Processes depth-wise soil data: missing values, outlier removal, scaling."""
    input_path, output_path = Path(input_dir), Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    depth_data = load_depth_data(input_path)
    
    for depth, df in depth_data.items():
        logger.info(f"\nProcessing: {depth} depth...")
        
        df.replace(["", " ", "NULL", "NaN", "nan"], np.nan, inplace=True)
        df = handle_missing_values(df)
        df = remove_outliers(df)
        df = scale_features(df)
        
        if df.isnull().sum().sum() > 0:
            logger.warning(f"Warning: {depth} file still contains NaNs after processing!")
        
        output_file = output_path / f"cleaned_depth_{depth}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Processed & saved: {output_file}")

# =========================== EXECUTE SCRIPT =========================== #
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "depthData"
    output_dir = base_dir / "cleanedData"
    
    logger.info(f"Input Directory: {input_dir}")
    logger.info(f"Output Directory: {output_dir}")
    
    preprocess_depth_data(str(input_dir), str(output_dir))
    logger.info("\n**All depth-wise files processed successfully!**")
