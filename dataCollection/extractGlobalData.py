import pandas as pd
import os
import logging
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/extractGlobalData.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def generate_depth_columns(depth: str) -> List[str]:
    """
    Dynamically generate column names for a given depth range.

    Args:
        depth (str): The depth range, e.g., '0-5cm', '5-15cm', etc.

    Returns:
        List[str]: A list of column names associated with the given depth.
    """
    base_features = [
        'bdod', 'cec', 'cfvo', 'clay', 'nitrogen', 'ocd', 
        'phh2o', 'sand', 'silt', 'soc', 'wv0010', 'wv0033', 'wv1500'
    ]

    return [f"{feature}_{depth}_{suffix}" 
            for feature in base_features 
            for suffix in ["Q0.05", "Q0.5", "Q0.95", "mean", "uncertainty"]]

def extract_depth_data(global_file_path: str, output_dir: str) -> None:
    """
    Extracts depth-wise data from a global soil dataset and saves each depth's data into separate CSV files.

    Args:
        global_file_path (str): Path to the global soil dataset CSV file.
        output_dir (str): Directory to save the depth-wise CSV files.
    """
    try:
        # Ensure the global dataset file exists
        global_file = Path(global_file_path)
        if not global_file.exists():
            logger.error(f"Global dataset file not found: {global_file_path}")
            raise FileNotFoundError(f"Dataset file {global_file_path} does not exist.")
        
        # Load dataset
        logger.info(f"Loading global dataset from {global_file_path}...")
        global_data = pd.read_csv(global_file_path)

        # Define depth levels
        depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']

        # Create output directory if it does not exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each depth level
        for depth in depths:
            columns = generate_depth_columns(depth)
            required_columns = ["latitude", "longitude"] + columns

            # Ensure all required columns exist
            missing_columns = [col for col in required_columns if col not in global_data.columns]
            if missing_columns:
                logger.warning(f"Skipping {depth} due to missing columns: {missing_columns}")
                continue

            # Extract and save data
            depth_data = global_data[required_columns]
            output_file = output_dir / f"depth_{depth}.csv"
            depth_data.to_csv(output_file, index=False)

            logger.info(f"Extracted and saved data for {depth} at {output_file}")

        logger.info("Depth-wise data extraction completed successfully.")

    except Exception as e:
        logger.error(f"Error extracting depth data: {str(e)}", exc_info=True)
        raise

# Example usage
if __name__ == "__main__":
    global_file_path = "data/globalSoilData.csv"  # Path to global dataset
    output_dir = "depthData"      # Directory for extracted files
    extract_depth_data(global_file_path, output_dir)
