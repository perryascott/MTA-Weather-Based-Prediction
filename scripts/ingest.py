import pandas as pd
import os
from datetime import timedelta

def build_mta_df(subway_data_path, bus_data_path, weather_data_path):

    # Read in hourly bus ridership data
    hourly_bus_df = pd.read_csv(bus_data_path)

    # Read in hourly subway ridership data
    hourly_subway_df = pd.read_csv(subway_data_path)

    # Convert 'hour' column to datetime for both datasets
    hourly_bus_df['hour'] = pd.to_datetime(hourly_bus_df['hour'])
    hourly_subway_df['hour'] = pd.to_datetime(hourly_subway_df['hour'])

    # For bus data
    hourly_bus_df['hour'] = pd.to_datetime(hourly_bus_df['hour'], utc=True)
    # hourly_bus_df['hour'] = hourly_bus_df['hour'].dt.tz_convert('America/New_York')

    # For subway data
    hourly_subway_df['hour'] = pd.to_datetime(hourly_subway_df['hour'], utc=True)
    # hourly_subway_df['hour'] = hourly_subway_df['hour'].dt.tz_convert('America/New_York')

    # Combine subway and bus hourly data
    hourly_subway_df['transportation'] = 'Subway'
    hourly_bus_df['transportation'] = 'Bus'

    # Convert 'hour' column to datetime for hourly data and remove timezone info
    hourly_bus_df['hour'] = pd.to_datetime(hourly_bus_df['hour']).dt.tz_localize(None)
    hourly_subway_df['hour'] = pd.to_datetime(hourly_subway_df['hour']).dt.tz_localize(None)

    weather_df = pd.read_csv(weather_data_path)

    weather_df['time'] = pd.to_datetime(weather_df['time'])

    # Calculate additional features
    weather_df['heat_index'] = weather_df['temp_c'] + 0.33 * weather_df['humidity'] - 0.70
    weather_df['wind_chill'] = 13.12 + 0.6215 * weather_df['temp_c'] - 11.37 * weather_df['wind_kph']**0.16 + 0.3965 * weather_df['temp_c'] * weather_df['wind_kph']**0.16
    weather_df['discomfort_index'] = 0.5 * (weather_df['temp_c'] + 61.0 + ((weather_df['temp_c'] - 68.0) * 1.2) + (weather_df['humidity'] * 0.094))
    weather_df['humidity_temperature_index'] = weather_df['humidity'] * weather_df['temp_c']

    # Find the earliest datetime in the weather data
    earliest_weather_datetime = weather_df['time'].min()

    # Cutoff the hourly ridership data before the earliest hour in the weather data
    hourly_subway_df = hourly_subway_df[hourly_subway_df['hour'] >= earliest_weather_datetime]
    hourly_bus_df = hourly_bus_df[hourly_bus_df['hour'] >= earliest_weather_datetime]

    # Rename columns for clarity
    weather_df = weather_df.rename(columns={
        'precip_in': 'Precipitation (in)',
        'pressure_in': 'Pressure (inHg)',
        'cloud': 'Cloud Cover (%)',
        'humidity': 'Relative Humidity (%)',
        'temp_f': 'Temperature (°F)',
        'wind_mph': 'Wind Speed (mph)',
        'gust_mph': 'Wind Gust (mph)',
        'vis_miles': 'Visibility (miles)'
    })

    return hourly_subway_df, hourly_bus_df, weather_df

    
# Add a 'season' column
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'
    
def get_combined_residuals_df(hourly_subway_df, hourly_bus_df, weather_df, subway_decomposition, bus_decomposition):

    # Combine ridership data, residuals, and weather data into a single DataFrame
    combined_df = pd.merge(
        hourly_subway_df[['hour', 'total_ridership']].rename(columns={'total_ridership': 'subway_ridership'}),
        pd.DataFrame({'hour': hourly_subway_df['hour'], 'subway_residual': subway_decomposition.resid}),
        on='hour'
    )

    combined_df = pd.merge(
        combined_df,
        hourly_bus_df[['hour', 'total_ridership']].rename(columns={'total_ridership': 'bus_ridership'}),
        on='hour'
    )

    combined_df = pd.merge(
        combined_df,
        pd.DataFrame({'hour': hourly_bus_df['hour'], 'bus_residual': bus_decomposition.resid}),
        on='hour'
    )

    combined_df = pd.merge(
        combined_df,
        weather_df,
        left_on='hour',
        right_on='time'
    )

    # Drop the redundant 'time' column
    combined_df = combined_df.drop('time', axis=1)

    combined_df['season'] = combined_df['hour'].dt.month.map(get_season)

    # Add an 'is_weekend' column
    combined_df['is_weekend'] = combined_df['hour'].dt.dayofweek.isin([5, 6]).astype(int)

    return combined_df