import matplotlib.pyplot as plt

def plot_city_aqi_trend(df, city):
    city_data = df[df["City"] == city]
    plt.figure(figsize=(10, 5))
    plt.plot(city_data.index, city_data["Predicted_AQI"], marker='o')
    plt.title(f"AQI Trend for {city}")
    plt.xlabel("Sample")
    plt.ylabel("Predicted AQI")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
