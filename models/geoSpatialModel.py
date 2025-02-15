import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import logging

class GeospatialSoilPredictor:
    def __init__(self, input_dir, logging_level=logging.INFO):
        """
        Initialize the Geospatial Soil Predictor with advanced features.
        
        Args:
            input_dir (str): Directory containing cleaned depth-wise soil data
            logging_level (int): Logging level for tracking prediction process
        """
        # Configure logging
        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load and prepare data
        self.input_dir = input_dir
        self.depth_data = self.load_cleaned_depth_data()
        self.kdtrees = self.build_kdtrees()
        self.scalers = self.prepare_scalers()
    
    def load_cleaned_depth_data(self):
        """
        Load and validate depth-wise soil data with comprehensive checks.
        
        Returns:
            dict: Validated depth-wise soil data
        """
        depth_files = [f for f in os.listdir(self.input_dir) if f.endswith(".csv")]
        depth_data = {}
        
        if not depth_files:
            raise ValueError("No CSV files found in the input directory")
        
        for file in depth_files:
            try:
                depth = file.replace("cleaned_depth_", "").replace(".csv", "")
                df = pd.read_csv(os.path.join(self.input_dir, file))
                
                # Validate required columns
                required_columns = ['latitude', 'longitude']
                if not all(col in df.columns for col in required_columns):
                    raise ValueError(f"Missing required columns in {file}")
                
                depth_data[depth] = df
                self.logger.info(f"Loaded data for depth {depth}: {len(df)} records")
            
            except Exception as e:
                self.logger.error(f"Error loading {file}: {e}")
        
        return depth_data
    
    def build_kdtrees(self):
        """
        Build efficient KD-Trees for nearest neighbor searches.
        
        Returns:
            dict: KD-Trees for each depth
        """
        kdtrees = {}
        for depth, df in self.depth_data.items():
            kdtrees[depth] = cKDTree(df[['latitude', 'longitude']].values)
        return kdtrees
    
    def prepare_scalers(self):
        """
        Prepare StandardScaler for each depth to normalize features.
        
        Returns:
            dict: Scalers for each depth
        """
        scalers = {}
        for depth, df in self.depth_data.items():
            scaler = StandardScaler()
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_columns = [col for col in numeric_columns if col not in ['latitude', 'longitude']]
            
            if numeric_columns:
                scaler.fit(df[numeric_columns])
                scalers[depth] = {
                    'scaler': scaler,
                    'columns': numeric_columns
                }
        return scalers
    
    def predict_soil_properties(self, latitude, longitude):
        """
        Predict soil properties with enhanced prediction methodology.
        
        Args:
            latitude (float): Geographical latitude
            longitude (float): Geographical longitude
        
        Returns:
            dict: Predicted soil properties for different depths
        """
        results = {}

        for depth, df in self.depth_data.items():
            try:
                # Check for exact location match
                exact_match = df[(df["latitude"] == latitude) & (df["longitude"] == longitude)]
                
                if not exact_match.empty:
                    results[depth] = exact_match.iloc[0].to_dict()
                    results[depth]["confidence"] = 1.0
                else:
                    # Advanced nearest neighbor prediction
                    dist, idx = self.kdtrees[depth].query([latitude, longitude], k=5)
                    nearest_neighbors = df.iloc[idx]
                    
                    # Prepare training data
                    X_train = nearest_neighbors[['latitude', 'longitude']]
                    y_train = nearest_neighbors.drop(columns=['latitude', 'longitude'])
                    
                    # Apply scaling if configured
                    if depth in self.scalers:
                        scaler_info = self.scalers[depth]
                        y_train_scaled = scaler_info['scaler'].transform(y_train[scaler_info['columns']])
                        y_train_scaled = pd.DataFrame(
                            y_train_scaled, 
                            columns=scaler_info['columns'], 
                            index=y_train.index
                        )
                    
                    # KNN Regression with distance-based weighting
                    knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance')
                    knn_model.fit(X_train, y_train_scaled if 'y_train_scaled' in locals() else y_train)
                    
                    # Predict and potentially inverse transform
                    predicted_values = knn_model.predict([[latitude, longitude]])[0]
                    if depth in self.scalers:
                        predicted_values = self.scalers[depth]['scaler'].inverse_transform(
                            predicted_values.reshape(1, -1)
                        )[0]
                    
                    # Compile results
                    results[depth] = dict(zip(y_train.columns, predicted_values))
                    results[depth]["confidence"] = max(0.1, 1 - dist.mean()/100)
            
            except Exception as e:
                self.logger.error(f"Prediction error for depth {depth}: {e}")
                results[depth] = {"error": str(e), "confidence": 0.0}
        
        return results
