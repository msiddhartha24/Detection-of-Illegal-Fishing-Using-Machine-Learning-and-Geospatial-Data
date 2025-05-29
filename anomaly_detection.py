def run_anomaly_detection(data):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy
    import math
    import geopandas as gpd
    import geodatasets
    from shapely.geometry import Point, Polygon
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.svm import SVR
    from sklearn.model_selection import train_test_split
    import numpy as np

    # ------------------ Your Existing Code Starts Here ------------------

    data.isnull().any()
    missing_values_columns = data.columns[data.isnull().any()].tolist()
    count_missing_values = []
    for i in missing_values_columns:
        count = data[i].isnull().sum()
        count_missing_values.append(count)

    data = data.interpolate(method='linear', axis=0).ffill().bfill()

    data_with_nan = data.copy()
    for col in ["LON", "LAT"]:
        data_with_nan.loc[data[col].isnull(), col] = None

    # Create one vessel sample
    data['MMSI'] = data['MMSI'].astype(int)
    track1 = data[data['MMSI'] == 367390380]

    year = pd.DatetimeIndex(track1['BaseDateTime']).year.tolist()
    month = pd.DatetimeIndex(track1['BaseDateTime']).month.tolist()
    day = pd.DatetimeIndex(track1['BaseDateTime']).day.tolist()
    hour = pd.DatetimeIndex(track1['BaseDateTime']).hour.tolist()
    minute = pd.DatetimeIndex(track1['BaseDateTime']).minute.tolist()

    track1['Year'] = year
    track1['month'] = month
    track1['day'] = day
    track1['hour'] = hour
    track1['minute'] = minute

    numeric_track1 = track1.select_dtypes(include=['number'])
    correlation = numeric_track1.corrwith(track1["SOG"])

    y = track1['SOG'].ravel()
    X = track1[['LAT', 'hour', 'Cargo', 'COG']]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=4)

    regressor = SVR(gamma='scale', C=100000, epsilon=1, degree=3)
    regressor.fit(x_train, y_train)
    yhat = regressor.predict(x_test)

    yhat = np.round(yhat, 1)
    predicted_vs_real = pd.DataFrame({'Actual': y_test, 'Predicted': yhat})
    predicted_vs_real['difference'] = predicted_vs_real['Actual'] - predicted_vs_real['Predicted']
    predicted_vs_real['Rounded Difference'] = np.round(predicted_vs_real['difference'].tolist(), 0)

    anamolies = predicted_vs_real[
        (predicted_vs_real['Rounded Difference'] >= 6) |
        (predicted_vs_real['Rounded Difference'] <= -6)
    ]

    x_test.reset_index(inplace=True)
    x_test.drop('index', axis=1, inplace=True)
    anamolies_indices = anamolies.index.tolist()

    # Update BaseDateTime format
    data['BaseDateTime'] = pd.to_datetime(data['BaseDateTime'])

    # Convert to GeoDataFrame for plotting
    geometry = [Point(xy) for xy in zip(data['LON'], data['LAT'])]
    gdf = gpd.GeoDataFrame(data, geometry=geometry)

    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

    # Save figure instead of plt.show()
    fig, ax = plt.subplots(figsize=(12, 16))
    world.plot(ax=ax, color='lightgrey')
    gdf.plot(ax=ax, marker='o', color='red', markersize=7)

    output_path = "anomaly_map.png"
    plt.title("Detected Ship Movement Anomalies")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    # ------------------ Your Existing Code Ends Here ------------------

    return output_path
