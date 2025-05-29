# iuu_detection.py

import pandas as pd
import folium
from geopy.distance import geodesic

def run_iuu_detection(data):

    #df = pd.read_csv(data)
    
    
    df = data.copy()
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
    df.sort_values(by=["MMSI", "BaseDateTime"], inplace=True)

    loitering_vessels = []
    for mmsi, group in df.groupby("MMSI"):
        slow = group[group['SOG'] < 0.5]
        if len(slow) >= 2:
            duration = (slow['BaseDateTime'].iloc[-1] - slow['BaseDateTime'].iloc[0]).total_seconds() / 3600
            if duration > 1:
                loitering_vessels.append((mmsi, slow.iloc[0]['LAT'], slow.iloc[0]['LON']))

    disappearing_vessels = []
    for mmsi, group in df.groupby("MMSI"):
        times = group['BaseDateTime'].sort_values().diff().dt.total_seconds().fillna(0)
        if any(times > 4 * 3600):
            idx = times.idxmax()
            row = group.loc[idx]
            disappearing_vessels.append((mmsi, row['LAT'], row['LON']))

    encountering_vessels = []
    mmsis = df['MMSI'].unique()
    for i in range(len(mmsis)):
        for j in range(i+1, len(mmsis)):
            ship1 = df[df['MMSI'] == mmsis[i]]
            ship2 = df[df['MMSI'] == mmsis[j]]
            merged = pd.merge_asof(ship1.sort_values('BaseDateTime'),
                                   ship2.sort_values('BaseDateTime'),
                                   on='BaseDateTime',
                                   direction='nearest',
                                   tolerance=pd.Timedelta('2min'),
                                   suffixes=('_1', '_2'))
            for _, row in merged.iterrows():
                if row['SOG_1'] < 1 and row['SOG_2'] < 1:
                    d = geodesic((row['LAT_1'], row['LON_1']), (row['LAT_2'], row['LON_2'])).km
                    if d < 1:
                        encountering_vessels.append((mmsis[i], mmsis[j], row['LAT_1'], row['LON_1']))
                        break

    restricted_zone = {
        'lat_min': df['LAT'].min() + 0.5,
        'lat_max': df['LAT'].min() + 1.0,
        'lon_min': df['LON'].min() + 0.5,
        'lon_max': df['LON'].min() + 1.0
    }
    restricted_zone_vessels = []
    for _, row in df.iterrows():
        if (restricted_zone['lat_min'] <= row['LAT'] <= restricted_zone['lat_max'] and
            restricted_zone['lon_min'] <= row['LON'] <= restricted_zone['lon_max']):
            restricted_zone_vessels.append((row['MMSI'], row['LAT'], row['LON']))

    m = folium.Map(location=[df['LAT'].mean(), df['LON'].mean()], zoom_start=6, tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", attr="OpenTopoMap")

    for mmsi, lat, lon in loitering_vessels:
        folium.Marker([lat, lon], icon=folium.Icon(color="blue"), popup=f"Loitering: {mmsi}").add_to(m)
    for mmsi, lat, lon in disappearing_vessels:
        folium.Marker([lat, lon], icon=folium.Icon(color="gray"), popup=f"Disappearing: {mmsi}").add_to(m)
    for mmsi1, mmsi2, lat, lon in encountering_vessels:
        folium.Marker([lat, lon], icon=folium.Icon(color="orange"), popup=f"Encounter: {mmsi1} & {mmsi2}").add_to(m)
    for mmsi, lat, lon in restricted_zone_vessels:
        folium.Marker([lat, lon], icon=folium.Icon(color="red"), popup=f"Restricted Zone: {mmsi}").add_to(m)

    output_html = "iuu_detection_map.html"
    m.save(output_html)
    return output_html
