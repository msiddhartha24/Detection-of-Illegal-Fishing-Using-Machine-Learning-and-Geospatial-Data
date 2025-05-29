import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import folium
from folium import plugins
import streamlit as st
from streamlit.components.v1 import html

def show_outlier_trajectories_map(csv_path):
    """
    Load AIS data, detect outlier vessel trajectories using DBSCAN,
    and display the outliers on a folium map within Streamlit.
    
    Args:
        csv_path (str): Path to the AIS CSV file.
    """
    # Load the AIS data
    df = pd.read_csv(csv_path)

    # Convert time column
    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])

    # Sort and group by MMSI
    df = df.sort_values(['MMSI', 'BaseDateTime'])
    trajectories = []

    # Resample: Take 10 equally spaced points for each vessel
    for mmsi, group in df.groupby('MMSI'):
        group = group.sort_values('BaseDateTime')
        if len(group) >= 10:
            resampled = group.iloc[np.linspace(0, len(group)-1, 10).astype(int)]
            lat_lon = resampled[['LAT', 'LON']].values.flatten()
            trajectories.append([mmsi] + list(lat_lon))

    # Create DataFrame
    columns = ['MMSI'] + [f'coord_{i}' for i in range(20)]  # 10 LAT, 10 LON
    route_df = pd.DataFrame(trajectories, columns=columns)

    # Normalize coordinates
    X = route_df.drop(columns=['MMSI'])
    X_scaled = StandardScaler().fit_transform(X)

    # Clustering using DBSCAN
    db = DBSCAN(eps=1.5, min_samples=3).fit(X_scaled)
    route_df['Cluster'] = db.labels_

    # Separate outliers
    outliers = route_df[route_df['Cluster'] == -1]
    normal_routes = route_df[route_df['Cluster'] != -1]

    # Print summary
    st.write(f"### Trajectory Clustering Summary")
    st.write(f"**Total Trajectories:** {len(route_df)}")
    st.write(f"**Normal Clusters:** {len(route_df['Cluster'].unique()) - (1 if -1 in route_df['Cluster'].unique() else 0)}")
    st.write(f"**Outlier Trajectories:** {len(outliers)}")

    # Create base map centered around average lat/lon
    map_center = [df['LAT'].mean(), df['LON'].mean()]
    m = folium.Map(location=map_center, zoom_start=5, tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", attr="OpenTopoMap")

    # Plot each outlier trajectory
    for idx, row in outliers.iterrows():
        mmsi = row['MMSI']
        lat_lon_coords = [(row[f'coord_{i}'], row[f'coord_{i+1}']) for i in range(0, 20, 2)]

        # Add line for trajectory
        folium.PolyLine(lat_lon_coords, color='red', weight=3.5, tooltip=f"MMSI: {mmsi} (Suspicious)").add_to(m)

        # Optional: Start and End markers
        folium.Marker(lat_lon_coords[0], icon=folium.Icon(color='green'), tooltip="Start").add_to(m)
        folium.Marker(lat_lon_coords[-1], icon=folium.Icon(color='red'), tooltip="End").add_to(m)

    # Display the map in Streamlit
    html(m._repr_html_(), height=600)
