from scripts.prediction_model import train_and_save_model

data_path = "data/aqi_dataset_full_5000.csv"
model_path = "models/rf_model.pkl"

train_and_save_model(data_path, model_path)
