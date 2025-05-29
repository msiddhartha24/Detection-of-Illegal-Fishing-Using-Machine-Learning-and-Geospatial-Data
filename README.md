
## 🌊 Illegal, Unreported, and Unregulated (IUU) Fishing Detection Using Machine Learning & Geospatial Data

This project focuses on detecting IUU (Illegal, Unreported, and Unregulated) fishing activities using a combination of machine learning techniques and geospatial analysis. By leveraging Automatic Identification System (AIS) data and satellite imagery, the system can monitor vessel movements, flag suspicious behavior, and support maritime authorities in enforcing sustainable fishing regulations.
 🔍Key Features

* Ship Detection using CNN: Detect vessels from satellite images using a trained Convolutional Neural Network.
* Rule-Based Detection: Identify IUU vessels based on maritime rules such as loitering, AIS signal gaps, restricted zone entry, and nighttime activity.
* Anomaly Detection: Visualize spatial anomalies in vessel movement patterns.
* Trajectory Clustering: Use DBSCAN to discover common vessel routes and highlight abnormal trajectories as potential outliers.
* Interactive Streamlit Interface: A user-friendly dashboard to upload AIS data, visualize maps, detect suspicious activity, and monitor flagged vessels.

-- 📊 Technologies Used

* Python, Pandas, NumPy
* Machine Learning (Logistic Regression, Clustering)
* Deep Learning (CNN for image-based ship detection)
* Folium & Streamlit (for visualization and interaction)
* Scikit-learn, Matplotlib, PIL, Geopy

---

📂Dataset

* AIS Data: Collected from maritime tracking systems, including vessel ID (MMSI), speed, position, and timestamps.
* Satellite Imagery: Sourced from open datasets like ShipsNet, used for ship detection training and validation.

---

🎯Goal

To provide an efficient, automated system capable of monitoring and flagging potential IUU fishing activity using open-access maritime data and intelligent algorithms—supporting efforts in sustainable fishing and marine conservation.

