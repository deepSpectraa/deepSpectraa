import requests
import time

def fetch_soil_data(latitude, longitude):
    """
    Fetches soil nutrient data for given latitude, longitude from SoilGrids API.
    """
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={longitude}&lat={latitude}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        if "properties" in data:
            properties = data["properties"]
            soil_info = {"latitude": latitude, "longitude": longitude}

            for layer in properties.get("layers", []):
                property_name = layer.get('name', 'unknown_property')
                if 'depths' in layer:
                    for depth in layer['depths']:
                        depth_label = depth.get("label", "unknown_depth")
                        for key, value in depth.get("values", {}).items():
                            column_name = f"{property_name}_{depth_label}_{key}"
                            soil_info[column_name] = value

            return soil_info
        else:
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {latitude}, {longitude}: {e}")
        return None
