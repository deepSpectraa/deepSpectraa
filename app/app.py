import streamlit as st
import pandas as pd
import os
import sys
import time
from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import logging

# Configure logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/soilAnalysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure proper module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.geoSpatialModel import GeospatialSoilPredictor

# Dictionary mapping technical property codes to user-friendly names with units
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

# Dictionary mapping statistical measures to user-friendly names
MEASURE_MAPPING = {
    'Q0.05': '5th Percentile',
    'Q0.5': 'Median',
    'Q0.95': '95th Percentile',
    'mean': 'Mean',
    'uncertainty': 'Uncertainty'
}

def format_property_name(technical_name):
    """
    Converts technical property names to user-friendly display names.
    
    Args:
        technical_name (str): Original property name (e.g., 'bdod_0-5cm_mean')
        
    Returns:
        str: Formatted property name with measurement type
    """
    try:
        if technical_name == 'confidence':
            return 'Confidence'
            
        parts = technical_name.split('_')
        if len(parts) < 2:
            return technical_name
            
        base_property = parts[0]
        measure = parts[-1]
        
        property_name = PROPERTY_MAPPING.get(base_property, base_property)
        measure_name = MEASURE_MAPPING.get(measure, measure)
        
        return f"{property_name} - {measure_name}"
    except Exception as e:
        logger.error(f"Error formatting property name: {e}")
        return technical_name

def get_location_name(lat, lon, retries=3, delay=2):
    """
    Fetches the location name from latitude & longitude coordinates.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        retries (int): Number of retry attempts
        delay (int): Delay between retries in seconds
    
    Returns:
        str: Location name or "Unknown Location" if not found
    """
    geolocator = Nominatim(user_agent="soil_nutrient_prediction")
    
    for attempt in range(retries):
        try:
            location = geolocator.reverse((lat, lon), exactly_one=True)
            if location and location.raw.get("address"):
                address = location.raw["address"]
                return (
                    address.get("city") or
                    address.get("town") or
                    address.get("village") or
                    address.get("state") or
                    address.get("country") or
                    "Unknown Location"
                )
        except GeocoderTimedOut:
            if attempt < retries - 1:  # Don't sleep on the last attempt
                time.sleep(delay)
    
    return "Unknown Location"

def preprocess_data_for_visualization(df):
    """
    Preprocesses the dataframe for visualization with improved handling of data types and missing values.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    try:
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Convert values to numeric, coercing errors to NaN
        df_processed['Value'] = pd.to_numeric(df_processed['Value'], errors='coerce')
        
        # Handle missing values
        df_processed = df_processed.dropna(subset=['Value'])
        
        # Normalize property names
        df_processed['Property'] = df_processed['Property'].astype(str)
        
        # Sort by depth and property
        depth_order = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
        df_processed['Depth'] = pd.Categorical(df_processed['Depth'], categories=depth_order, ordered=True)
        df_processed = df_processed.sort_values(['Depth', 'Property'])
        
        return df_processed
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def create_heatmap(df_combined):
    """
    Creates an enhanced heatmap with formatted property names and improved data processing.
    
    Args:
        df_combined (pd.DataFrame): Combined dataframe with columns [Depth, Property, Value]
        
    Returns:
        plotly.graph_objects.Figure: Heatmap figure object
    """
    try:
        # Create a copy to avoid modifying the original
        df = df_combined.copy()
        
        # Ensure numeric values
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        # Format property names more concisely
        df['Property'] = df['Property'].apply(lambda x: x.split('_')[0] if '_' in x else x)
        df['Property'] = df['Property'].apply(lambda x: PROPERTY_MAPPING.get(x, x))
        
        # Sort depths in correct order
        depth_order = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
        df['Depth'] = pd.Categorical(df['Depth'], categories=depth_order, ordered=True)
        
        # Create pivot table with sorted depths
        pivot_df = df.pivot_table(
            values='Value',
            index='Depth',
            columns='Property',
            aggfunc='mean'
        ).reindex(depth_order)
        
        # Calculate value ranges for better color scaling
        vmin = df['Value'].min()
        vmax = df['Value'].max()
        
        # Create heatmap with improved configuration
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='RdYlBu_r',
            zmin=vmin,
            zmax=vmax,
            hoverongaps=False,
            hovertemplate=(
                'Depth: %{y}<br>' +
                'Property: %{x}<br>' +
                'Value: %{z:.2f}<extra></extra>'
            )
        ))
        
        # Update layout with improved configurations
        fig.update_layout(
            title={
                'text': 'Soil Properties Heatmap Across Depths',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 18}
            },
            xaxis={
                'title': 'Soil Properties',
                'tickangle': 45,
                'tickfont': {'size': 10},
                'title_font': {'size': 14}
            },
            yaxis={
                'title': 'Depth',
                'tickfont': {'size': 12},
                'title_font': {'size': 14}
            },
            height=600,
            width=1000,
            margin={'l': 70, 'r': 50, 't': 80, 'b': 150},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Add colorbar title
        fig.update_traces(
            colorbar={
                'title': 'Value',
                'titleside': 'right',
                'titlefont': {'size': 12}
            }
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        st.error("Error creating heatmap. Please check your data.")
        return None

def create_radar_chart(df_combined):
    """Creates an enhanced radar chart with formatted property names."""
    try:
        # Preprocess data
        df_processed = preprocess_data_for_visualization(df_combined)
        
        # Format property names
        df_processed['Property'] = df_processed['Property'].apply(lambda x: 
            PROPERTY_MAPPING.get(x.split('_')[0], x.split('_')[0]))
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set3
        
        for idx, depth in enumerate(df_processed['Depth'].unique()):
            depth_data = df_processed[df_processed['Depth'] == depth]
            
            fig.add_trace(go.Scatterpolar(
                r=depth_data['Value'],
                theta=depth_data['Property'],
                name=depth,
                fill='toself',
                line=dict(color=colors[idx % len(colors)]),
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, df_processed['Value'].max()],
                    showline=True,
                    linewidth=1,
                    gridcolor='rgba(0,0,0,0.1)'
                )
            ),
            showlegend=True,
            title="Radar Chart of Soil Properties by Depth",
            height=600
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating radar chart: {e}")
        st.error("Error creating radar chart. Please check your data.")
        return None

def create_box_plots(df_combined):
    """Creates enhanced box plots with formatted property names."""
    try:
        # Preprocess data
        df_processed = preprocess_data_for_visualization(df_combined)
        
        # Format property names
        df_processed['Property'] = df_processed['Property'].apply(lambda x: 
            PROPERTY_MAPPING.get(x.split('_')[0], x.split('_')[0]))
        
        fig = px.box(
            df_processed,
            x='Property',
            y='Value',
            color='Depth',
            title='Distribution of Soil Properties Across Depths',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            xaxis_title="Soil Properties",
            yaxis_title="Value",
            height=600,
            showlegend=True,
            legend_title="Depth"
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
    except Exception as e:
        logger.error(f"Error creating box plots: {e}")
        st.error("Error creating box plots. Please check your data.")
        return None

def main():
    # Page Configuration
    st.set_page_config(
        page_title="Soil Nutrient Prediction",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stTitle {
            color: #2c3e50;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .stSubheader {
            color: #34495e;
        }
        .stButton>button {
            background-color: #27ae60;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #219a52;
        }
        .property-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # App Title
    st.title("🌍 deepSpectraa")
    st.markdown("##### Analyze Soil at Key Depths and See Beyond the Surface")
    
    try:
        # Data Path Configuration
        DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataProcessing/cleanedData"))
        if not os.path.exists(DATA_PATH):
            st.error("Data directory not found. Please check the path configuration.")
            st.stop()
        
        # Initialize Soil Predictor Model
        soil_predictor = GeospatialSoilPredictor(DATA_PATH)
        
        # Initialize Session State Variables
        if "selected_lat" not in st.session_state:
            st.session_state.selected_lat = 22.5726
        if "selected_lon" not in st.session_state:
            st.session_state.selected_lon = 88.3639
        if "selected_location_name" not in st.session_state:
            st.session_state.selected_location_name = "Kolkata, India"
        if "map_zoom" not in st.session_state:
            st.session_state.map_zoom = 5
        if "prediction_data" not in st.session_state:
            st.session_state.prediction_data = None
        
        # Create Tabs
        tab1, tab2 = st.tabs(["📍 Soil Prediction", "📊 Soil Visualization"])
        
        with tab1:
            # Map Section
            st.subheader("🗺️ Select Location")
            
            m = folium.Map(
                location=[st.session_state.selected_lat, st.session_state.selected_lon],
                zoom_start=st.session_state.map_zoom,
                control_scale=True
            )
            
            folium.Marker(
                location=[st.session_state.selected_lat, st.session_state.selected_lon],
                popup=f"{st.session_state.selected_location_name}\n{st.session_state.selected_lat:.5f}, {st.session_state.selected_lon:.5f}",
                icon=folium.Icon(color="red"),
            ).add_to(m)
            
            map_data = st_folium(m, width="100%", height=400, key="folium_map")
            
            # Handle map interactions
            if map_data and map_data.get("last_clicked"):
                clicked_lat = map_data["last_clicked"]["lat"]
                clicked_lon = map_data["last_clicked"]["lng"]
                
                if (clicked_lat != st.session_state.selected_lat) or (clicked_lon != st.session_state.selected_lon):
                    st.session_state.selected_lat = clicked_lat
                    st.session_state.selected_lon = clicked_lon
                    st.session_state.selected_location_name = get_location_name(clicked_lat, clicked_lon)
                    st.session_state.map_zoom = map_data["zoom"]
                    st.rerun()
            
            # Coordinate inputs
            col1, col2 = st.columns(2)
            with col1:
                lat = st.number_input("Latitude", value=st.session_state.selected_lat, format="%.6f")
            with col2:
                lon = st.number_input("Longitude", value=st.session_state.selected_lon, format="%.6f")
            
            if lat != st.session_state.selected_lat or lon != st.session_state.selected_lon:
                st.session_state.selected_lat = lat
                st.session_state.selected_lon = lon
                st.session_state.selected_location_name = get_location_name(lat, lon)
            
            st.info(f"📍 Selected Location: {st.session_state.selected_location_name} ({st.session_state.selected_lat:.5f}, {st.session_state.selected_lon:.5f})")
            
            # Prediction Section
            if st.button("🔍 Predict Soil Nutrients"):
                with st.spinner("🌱 Analyzing soil properties..."):
                    try:
                        prediction = soil_predictor.predict_soil_properties(lat, lon)
                        
                        if prediction and isinstance(prediction, dict):
                            st.success("Prediction completed successfully!")
                            st.markdown("## 🌱 Soil Nutrient Prediction Results")
                            
                            combined_df = []
                            sorted_depths = sorted(prediction.keys(), key=lambda x: int(x.split('-')[0]))
                            
                            for depth in sorted_depths:
                                values = prediction[depth]
                                
                                with st.expander(f"Depth: {depth}", expanded=True):
                                    st.markdown(f"**Prediction Confidence:** `{'{:.2f}'.format(values.get('confidence', 0) * 100)}%`")
                                    
                                    df_depth = pd.DataFrame(list(values.items()), columns=["Property", "Value"])
                                    df_depth = df_depth[df_depth["Property"] != "confidence"]
                                    
                                    df_depth_display = df_depth.copy()
                                    df_depth_display["Property"] = df_depth_display["Property"].apply(format_property_name)
                                    
                                    st.dataframe(
                                        df_depth_display.set_index("Property"),
                                        height=300,
                                        use_container_width=True
                                    )
                                
                                df_depth.insert(0, "Depth", depth)
                                combined_df.append(df_depth)
                            
                            # Store the combined data in session state for visualization
                            st.session_state.combined_df = pd.concat(combined_df, ignore_index=True)
                            st.session_state.prediction_data = prediction  # Store raw prediction data
                            
                    except Exception as e:
                        logger.error(f"Prediction error: {str(e)}")
                        st.error(f"Prediction Error: {str(e)}")
                        st.info("Please try again with a different location or check your connection.")
        
        with tab2:
            st.subheader("📊 Data Visualization")
            
            if 'combined_df' in st.session_state and st.session_state.combined_df is not None:
                try:
                    # Create a container for visualization controls
                    viz_controls = st.container()
                    
                    with viz_controls:
                        # Visualization Type Selector
                        viz_type = st.selectbox(
                            "Select Visualization Type",
                            ["Heatmap", "Radar Chart", "Box Plots"],
                            help="Choose different visualization types to analyze the soil properties"
                        )
                        
                        # Create visualization container
                        viz_container = st.container()
                        
                        with viz_container:
                            # Display Selected Visualization with error handling
                            if viz_type == "Heatmap":
                                fig = create_heatmap(st.session_state.combined_df)
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True)
                                    with st.expander("📝 Understanding the Heatmap"):
                                        st.markdown("""
                                            The heatmap provides a color-coded visualization of soil properties across different depths:
                                            - Darker colors indicate higher values
                                            - Lighter colors represent lower values
                                            - Hover over cells to see exact values
                                            - X-axis shows different soil properties
                                            - Y-axis shows soil depths
                                        """)
                                
                            elif viz_type == "Radar Chart":
                                fig = create_radar_chart(st.session_state.combined_df)
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True)
                                    with st.expander("📝 Understanding the Radar Chart"):
                                        st.markdown("""
                                            The radar chart shows the relationship between different soil properties:
                                            - Each color represents a different depth
                                            - Distance from center indicates property value
                                            - Shape helps compare multiple properties simultaneously
                                            - Hover over points to see exact values
                                        """)
                                
                            elif viz_type == "Box Plots":
                                fig = create_box_plots(st.session_state.combined_df)
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True)
                                    with st.expander("📝 Understanding the Box Plots"):
                                        st.markdown("""
                                            Box plots show the statistical distribution of soil properties:
                                            - Box shows quartiles (25th to 75th percentile)
                                            - Line in box shows median
                                            - Whiskers show range of typical values
                                            - Points show outliers
                                            - Different colors represent different depths
                                        """)
                
                except Exception as e:
                    logger.error(f"Visualization error: {str(e)}")
                    st.error("An error occurred while creating the visualization. Please try again.")
            else:
                st.info("Hey! I think you forgot something please predict soil nutrients first to view visualizations!")
                st.markdown("""
                    #### How to get Visualization:
                    1. Select a desired location on the map or enter your choice coordinates
                    2. Click the "Predict Soil Nutrients" button to get useful soil data on that particular lat long
                    3. After following point 1 and 2 return to this tab to view visualizations :))
                """)
        
     # Add footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666;'>
                <p>© 2025 deepSpectra | Version 1.0.0</p>
                <p>
                    <a href="https://github.com/deepSpectraa/deepSpectraa/tree/main/docs" target="_blank">Documentation</a> |
                    <a href="https://github.com/deepSpectraa/deepSpectraa/issues/new/choose" target="_blank">Report Issues</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("""
            An unexpected error occurred. Want to resolve! if Yes then please try:
            1. Refreshing the page
            2. Clearing your browser cache
            3. If the problem still persists, contact support
        """)

if __name__ == "__main__":
    main()
