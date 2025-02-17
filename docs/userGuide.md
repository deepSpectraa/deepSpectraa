# deepSpectra User Guide

## Overview

deepSpectra is a sophisticated soil analysis application that provides detailed insights into soil properties at various depths. This guide will walk you through all aspects of using the application effectively.

## Getting Started

### Application Interface

The application is divided into two main tabs:
1. Soil Prediction
2. Soil Visualization

### Soil Prediction Tab

#### Map Navigation
- The interactive map allows you to select any location worldwide
- Click on any point to set coordinates
- The selected location's name will automatically appear below the map
- You can also manually input coordinates using the latitude and longitude fields

#### Making Predictions
1. Select a location using either method above
2. Click the "🔍 Predict Soil Nutrients" button
3. The system will analyze the following soil properties at various depths:
   - Bulk Density (cg/cm³)
   - Cation Exchange Capacity (mmol(c)/kg)
   - Coarse Fragments Volumetric (cm³/dm³)
   - Clay Content (g/kg)
   - Nitrogen Content (cg/kg)
   - Organic Carbon Density (hg/m³)
   - Soil pH (pHx10)
   - Sand Content (g/kg)
   - Silt Content (g/kg)
   - Soil Organic Carbon (dg/kg)
   - Water Retention at various pressures (10kPa, 33kPa, 1500kPa)

### Soil Visualization Tab

#### Available Visualizations

1. **Heatmap**
   - Shows soil properties across different depths
   - Color intensity indicates property values
   - Hover over cells for exact values

2. **Radar Chart**
   - Displays relationship between different properties
   - Each depth layer shown in different colors
   - Enables easy comparison across depths

3. **Box Plots**
   - Shows statistical distribution of properties
   - Displays quartiles, median, and outliers
   - Color-coded by depth

## Understanding Results

### Prediction Confidence
- Each prediction includes a confidence score
- Higher percentages indicate more reliable predictions
- Consider local conditions when interpreting results

### Depth Analysis
The system analyzes soil at six standard depths:
- 0-5cm
- 5-15cm
- 15-30cm
- 30-60cm
- 60-100cm
- 100-200cm

### Interpreting Values

#### Key Indicators

1. **Soil pH**
   - Values are shown in pHx10 format
   - Divide displayed value by 10 for actual pH
   - Optimal range: 5.5-7.5 (55-75 in displayed values)

2. **Organic Carbon**
   - Higher values indicate better soil health
   - Consider local climate when interpreting

3. **Water Retention**
   - Three pressure points provide soil water characteristics
   - Higher values indicate better water holding capacity

## Troubleshooting

### Common Issues

1. **Map Not Loading**
   - Check internet connection
   - Try refreshing the page
   - Clear browser cache if needed

2. **Prediction Errors**
   - Verify coordinates are within valid ranges
   - Ensure no special characters in input fields
   - Try a nearby location if issues persist

3. **Visualization Problems**
   - Make sure to run prediction first
   - Switch between visualization types
   - Refresh page if graphics don't render

### Error Messages

- "Prediction Error": Check coordinates and try again
- "Data directory not found": Contact support
- "An unexpected error occurred": Follow the three-step troubleshooting process

## Best Practices

1. **Location Selection**
   - Use precise coordinates when available
   - Verify location name matches expected area
   - Consider seasonal variations

2. **Data Interpretation**
   - Compare results with local soil tests when possible
   - Consider regional climate and geology
   - Use visualizations to identify patterns

3. **Regular Use**
   - Monitor changes over time
   - Document unusual readings
   - Compare seasonal variations

## Additional Resources

- GitHub Repository: [deepSpectra GitHub](https://github.com/deepSpectraa/deepSpectraa.git)
- Issue Tracking: [Report Issues](https://github.com/deepSpectraa/deepSpectraa/issues)
- Documentation: [Full Documentation](./docs/README.md)

## Support

For additional support:
1. Check the documentation
2. Submit issues on GitHub
3. Contact the development team
