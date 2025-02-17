# deepSpectra API Reference

## Core Classes

### GeospatialSoilPredictor

The main class handling soil property predictions.

```python
class GeospatialSoilPredictor:
    def __init__(self, data_path: str):
        """
        Initialize the soil predictor with data path.
        
        Args:
            data_path (str): Path to the cleaned data directory
        """

    def predict_soil_properties(self, latitude: float, longitude: float) -> dict:
        """
        Predict soil properties for given coordinates.
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            
        Returns:
            dict: Predicted soil properties at different depths
        """
```

## Utility Functions

### Location Services

```python
def get_location_name(lat: float, lon: float, retries: int = 3, delay: int = 2) -> str:
    """
    Get location name from coordinates.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        retries (int): Number of retry attempts
        delay (int): Delay between retries in seconds
        
    Returns:
        str: Location name or "Unknown Location"
    """
```

### Data Processing

```python
def preprocess_data_for_visualization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess dataframe for visualization.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
```

### Visualization Functions

```python
def create_heatmap(df_combined: pd.DataFrame) -> go.Figure:
    """
    Create heatmap visualization.
    
    Args:
        df_combined (pd.DataFrame): Combined dataframe with columns [Depth, Property, Value]
        
    Returns:
        plotly.graph_objects.Figure: Heatmap figure
    """

def create_radar_chart(df_combined: pd.DataFrame) -> go.Figure:
    """
    Create radar chart visualization.
    
    Args:
        df_combined (pd.DataFrame): Combined dataframe
        
    Returns:
        plotly.graph_objects.Figure: Radar chart figure
    """

def create_box_plots(df_combined: pd.DataFrame) -> go.Figure:
    """
    Create box plot visualization.
    
    Args:
        df_combined (pd.DataFrame): Combined dataframe
        
    Returns:
        plotly.graph_objects.Figure: Box plot figure
    """
```

## Constants

### Property Mapping

```python
PROPERTY_MAPPING = {
    'bdod': 'Bulk Density (cg/cm³)',
    'cec': 'Cation Exchange Capacity (mmol(c)/kg)',
    'cfvo': 'Coarse Fragments Volumetric (cm³/dm³)',
    'clay': 'Clay Content (g/kg)',
    'nitrogen': 'Nitrogen Content (cg/kg)',
    'ocd': 'Organic Carbon Density (hg/m³)',
    'phh2o': 'Soil pH (pHx10)',
    'sand': 'Sand Content (g/kg)',
    'silt': 'Silt Content (g/kg)',
    'soc': 'Soil Organic Carbon (dg/kg)',
    'wv0010': 'Water Retention at 10 kPa',
    'wv0033': 'Water Retention at 33 kPa',
    'wv1500': 'Water Retention at 1500 kPa'
}
```

### Measure Mapping

```python
MEASURE_MAPPING = {
    'Q0.05': '5th Percentile',
    'Q0.5': 'Median',
    'Q0.95': '95th Percentile',
    'mean': 'Mean',
    'uncertainty': 'Uncertainty'
}
```

## Data Structures

### Prediction Output Format

```python
{
    "0-5cm": {
        "bdod": float,
        "cec": float,
        "cfvo": float,
        "clay": float,
        "nitrogen": float,
        "ocd": float,
        "phh2o": float,
        "sand": float,
        "silt": float,
        "soc": float,
        "wv0010": float,
        "wv0033": float,
        "wv1500": float,
        "confidence": float
    },
    "5-15cm": {...},
    "15-30cm": {...},
    "30-60cm": {...},
    "60-100cm": {...},
    "100-200cm": {...}
}
```

## Error Handling

### Logging Configuration

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/soilAnalysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
```

### Error Types

1. **GeocoderTimedOut**: Location service timeout
2. **DataDirectoryError**: Missing or invalid data directory
3. **PredictionError**: Error during soil property prediction
4. **VisualizationError**: Error creating visualizations

## Dependencies

- streamlit
- pandas
- folium
- geopy
- plotly
- numpy
- logging

## Configuration

### Page Configuration

```python
st.set_page_config(
    page_title="Soil Nutrient Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### Default Settings

```python
DEFAULT_COORDINATES = {
    "latitude": 22.5726,
    "longitude": 88.3639,
    "location": "Kolkata, India",
    "zoom": 5
}
```
