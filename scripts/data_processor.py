import pandas as pd

def load_and_process_data(file_path):
    df = pd.read_csv(file_path)

    df.dropna(inplace=True)

    # Convert pollutant/weather columns to numeric
    columns_to_convert = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'WindSpeed']
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=columns_to_convert, inplace=True)
    return df
