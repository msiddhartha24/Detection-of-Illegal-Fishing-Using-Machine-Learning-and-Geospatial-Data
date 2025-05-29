import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from PIL import Image
import anomaly_detection
from ship_detection import detect_ships_from_image
from iuu_detection import run_iuu_detection
from iuu_vessels import run_iuu_vessels_detection
from trajectory_outliers import show_outlier_trajectories_map
import os

# Page Configuration
st.set_page_config(page_title="IUU Fishing Detection", layout="wide")

# Title
st.title("🚢 Detection of Illegal, Unreported and Unregulated (IUU) Fishing")

# Intro Text
st.markdown("""
Welcome to our platform that **detects IUU fishing activities** using a combination of **AIS data**, **satellite imagery**, **machine learning models**, and **geospatial rule-based techniques**.
""")

# Sidebar
st.sidebar.title("🔹 Navigation")
page = st.sidebar.radio("Go to:", [
    "📘 Project Overview",
    "📊 Model Visualizations",
    "🛳️ Detected Ships (CNN)",
    "📂 Upload & Explore (Rule-Based IUU Detection)",
    "🔍 Anomaly Detection",
    "🌍 IUU Rules-Based Detection (Map)",
    "🧭 Trajectory Clustering (DBSCAN)"
])

# Main Pages
# ---------------------------------------------------------------
if page == "📘 Project Overview":
    st.header("📘 Project Summary")
    st.markdown("""
    - **Data Sources**: AIS Data + Satellite Images
    - **Models**: 
        - CNN Model for Ship Detection
        - Logistic Regression for Vessel IUU Classification
    - **Detection Techniques**: Loitering, Encountering, AIS Gaps, Restricted Zones
    - **Visualization**: Interactive Folium Maps
    - **Metrics**: Accuracy, Precision, Recall, F1 Score, AUC-ROC
    """)

# ---------------------------------------------------------------
elif page == "📊 Model Visualizations":
    st.header("📊 Model Performance")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
    with col2:
        st.image("roc_curve.png", caption="ROC Curve", use_container_width=True)
    with col3:
        st.image("metrics_bar.png", caption="Performance Metrics", use_container_width=True)

# ---------------------------------------------------------------
elif page == "🛳️ Detected Ships (CNN)":
    st.header("🛳️ Upload a Satellite Image for Ship Detection")
    uploaded_image = st.file_uploader("Upload Image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        with open("uploaded_satellite_image.png", "wb") as f:
            f.write(uploaded_image.read())

        with st.spinner("Detecting ships..."):
            result_path = detect_ships_from_image("uploaded_satellite_image.png")

        st.success("✅ Ship detection completed successfully!")
        st.image(result_path, caption="Detected Ships", use_container_width=True)

# ---------------------------------------------------------------
elif page == "📂 Upload & Explore (Rule-Based IUU Detection)":
    st.header("📂 Upload Your AIS Data")
    uploaded = st.file_uploader("Upload AIS CSV file", type=["csv"])

    if uploaded:
        data = pd.read_csv(uploaded)
        st.dataframe(data.head())

        if st.button("Run Rule-Based Detection"):
            with st.spinner("Processing uploaded data..."):
                iuu_df, _ = run_iuu_vessels_detection(data)

            if not iuu_df.empty:
                st.success("✅ IUU Vessels Detected Successfully!")
                iuu_df.to_csv("IUU_detected_vessels.csv", index=False)
                st.dataframe(iuu_df)

                st.subheader("🗺️ Vessel Path Map")
                m = folium.Map(location=[iuu_df['LAT'].mean(), iuu_df['LON'].mean()], zoom_start=6, tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", attr="OpenTopoMap")

                for mmsi in iuu_df['MMSI'].unique():
                    track = data[data['MMSI'] == mmsi].sort_values('BaseDateTime')
                    coords = list(zip(track['LAT'], track['LON']))
                    if len(coords) > 1:
                        folium.PolyLine(coords, color='blue', weight=2.5, tooltip=f"MMSI: {mmsi}").add_to(m)
                        folium.Marker(coords[0], icon=folium.Icon(color='green'), tooltip="Start").add_to(m)
                        folium.Marker(coords[-1], icon=folium.Icon(color='red'), tooltip="End").add_to(m)

                folium_static(m, height=600)
            else:
                st.warning("No suspicious vessels detected in the uploaded dataset.")

# ---------------------------------------------------------------
elif page == "🔍 Anomaly Detection":
    st.header("🔍 Anomaly Detection Based on Vessel Behavior")
    anomaly_file = st.file_uploader("Upload CSV for Anomaly Detection", type=["csv"])

    if anomaly_file:
        anomaly_data = pd.read_csv(anomaly_file)
        image_path = anomaly_detection.run_anomaly_detection(anomaly_data)
        st.image(image_path, caption="Detected Anomalies", use_column_width=True)

# ---------------------------------------------------------------
elif page == "🌍 IUU Rules-Based Detection (Map)":
    st.header("🌍 Visualize IUU Detections Using Geospatial Rules")
    iuu_csv = st.file_uploader("Upload AIS CSV File for Rule-Based Map", type=["csv"])

    if iuu_csv:
        df = pd.read_csv(iuu_csv)
        with st.spinner("Applying detection rules..."):
            map_path = run_iuu_detection(df)

        st.success("✅ Detection complete! Interactive map generated.")
        st.components.v1.html(open(map_path, 'r').read(), height=600)

# ---------------------------------------------------------------
elif page == "🧭 Trajectory Clustering (DBSCAN)":
    st.header("🧭 Vessel Trajectory Outlier Detection (DBSCAN)")
    traj_file = st.file_uploader("Upload AIS CSV for Trajectory Analysis", type=["csv"])

    if traj_file:
        with st.spinner("Analyzing vessel movements..."):
            show_outlier_trajectories_map(traj_file)

# ---------------------------------------------------------------

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Developed for IUU Fishing Detection Project")
