import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_and_save_model(data_path, model_path):
    df = pd.read_csv(data_path)

    X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'WindSpeed']]
    y = df['AQI'] if 'AQI' in df.columns else df['Predicted_AQI']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, model_path)
    return model
