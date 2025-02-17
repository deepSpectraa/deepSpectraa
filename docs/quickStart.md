# deepSpectra Quickstart Guide

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/deepSpectraa/deepSpectraa.git
cd deepSpectra
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up the data directory:
```bash
mkdir -p dataProcessing/cleanedData
```

## Running the Application

1. Start the Streamlit server:
```bash
streamlit run app.py
```

2. Open your browser and navigate to:
```
http://localhost:8501
```

## Quick Usage Guide

### 1. Basic Soil Analysis

```python
# Example coordinates for New Delhi, India
latitude = 28.6139
longitude = 77.2090

# Get soil properties
soil_predictor = GeospatialSoilPredictor("path/to/data")
results = soil_predictor.predict_soil_properties(latitude, longitude)
```

### 2. Visualization Examples

```python
import plotly.express as px
from utils.visualization import create_heatmap, create_radar_chart

# Create heatmap
fig_heatmap = create_heatmap(your_data)
fig_heatmap.show()

# Create radar chart
fig_radar = create_radar_chart(your_data)
fig_radar.show()
```

## Basic Features

1. **Soil Property Prediction**
   - Select location on map
   - Click "Predict Soil Nutrients"
   - View detailed results

2. **Data Visualization**
   - Heatmap view
   - Radar chart
   - Box plots

3. **Location Services**
   - Map-based selection
   - Coordinate input
   - Automatic location naming

## Customization

### Changing Default Location

```python
# In app.py
st.session_state.selected_lat = your_latitude
st.session_state.selected_lon = your_longitude
```

### Modifying Visualization Styles

```python
# Custom color scheme for heatmap
fig.update_traces(
    colorscale='Viridis',  # or any other Plotly colorscale
    showscale=True
)
```

## Common Operations

### 1. Analyzing a Location

```python
# Step 1: Select location (via map or coordinates)
# Step 2: Click "Predict Soil Nutrients"
# Step 3: View results in expandable sections
```

### 2. Comparing Depths

```python
# Step 1: Make prediction
# Step 2: Go to Visualization tab
# Step 3: Select "Radar Chart" to compare depths
```

### 3. Exporting Data

```python
# Data is available in pandas DataFrame format
# Can be exported using:
df.to_csv("soil_analysis.csv")
```

## Troubleshooting

### Common Issues

1. **Map Not Loading**
```python
# Check internet connection
# Verify Folium installation:
pip install folium --upgrade
```

2. **Prediction Errors**
```python
# Verify coordinate format:
latitude = float(latitude)
longitude = float(longitude)
```

3. **Visualization Issues**
```python
# Clear Streamlit cache:
st.cache_data.clear()
```

## Next Steps

1. Check the full [User Guide](userGuide.md)
2. Explore the [API Reference](apiReference.md)
3. Join our [Community](https://github.com/deepSpectraa/deepSpectraa/discussions)

## Getting Help

- Submit issues on GitHub
- Check documentation
- Contact support team

Remember to check the full documentation for detailed information about all features and capabilities.
