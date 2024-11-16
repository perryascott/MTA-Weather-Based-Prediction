import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.filter import filter_df_by_hour_range

def analyze_weather_impact(combined_df, lag=0, start_date=None, end_date=None, hour_range=None, is_weekend=None, plot=False):
    
    # Shift weather data by the specified lag
    combined_df_lagged = combined_df.copy()
    combined_df_lagged['hour'] = combined_df_lagged['hour'] + pd.Timedelta(hours=lag)

    # Filter data for the specified date range
    if start_date and end_date:
        start_month, start_day = map(int, start_date.split('-'))
        end_month, end_day = map(int, end_date.split('-'))
        
        def is_date_in_range(date):
            month, day = date.month, date.day
            if start_month <= end_month:
                return (start_month, start_day) <= (month, day) <= (end_month, end_day)
            else:
                return (month, day) >= (start_month, start_day) or (month, day) <= (end_month, end_day)
        
        combined_df_lagged = combined_df_lagged[combined_df_lagged['hour'].apply(is_date_in_range)]

    # Filter data for the specified hour range
    if hour_range:
        combined_df_lagged = filter_df_by_hour_range(combined_df_lagged, hour_range)

    # Filter data for weekday/weekend
    if is_weekend is not None:
        combined_df_lagged = combined_df_lagged[combined_df_lagged['is_weekend'] == is_weekend]

    # List of weather variables to analyze
    weather_vars = ['Temperature (°F)', 'Precipitation (in)', 'Relative Humidity (%)',
                    'Pressure (inHg)', 'Cloud Cover (%)']

    results = []
    for mode in ['subway', 'bus']:
        correlations = combined_df_lagged[weather_vars + [f'{mode}_residual']].corr()[f'{mode}_residual'].drop(f'{mode}_residual')
        for var, corr in correlations.items():
            result = {
                'Transportation Type': mode.capitalize(),
                'Weather Variable': var,
                'Correlation': corr
            }
            
            if is_weekend is not None:
                result['Day Type'] = 'Weekend' if is_weekend else 'Weekday'
            
            if 'season' in combined_df_lagged.columns:
                result['Season'] = combined_df_lagged['season'].iloc[0]
            
            if hour_range:
                result['Hour Segment'] = f'{hour_range[0]:02d}-{hour_range[1]:02d}'
                if 'season' in combined_df_lagged.columns:
                    result['Season_Hour'] = f'{combined_df_lagged["season"].iloc[0]}_{hour_range[0]:02d}-{hour_range[1]:02d}'
            
            results.append(result)

    correlation_df = pd.DataFrame(results)

    # Visualization code 
    if plot:
        plt.figure(figsize=(14, 10)) 
        sns.heatmap(correlation_df.pivot(index='Weather Variable', columns='Transportation Type', values='Correlation'), 
                    annot=True, cmap='coolwarm', center=0, annot_kws={"size": 16})  # Increased annotation size
        title = f'Correlation between Weather Variables and MSTL Residuals of Ridership'
        if start_date and end_date:
            title += f'\n{start_date} to {end_date} (2022, 2023, 2024)'
        if hour_range:
            title += f'\nHour range: {hour_range}'
        if is_weekend is not None:
            title += f'\nDay Type: {"Weekend" if is_weekend else "Weekday"}'
        plt.title(title, fontsize=16)  
        plt.xlabel('Transportation Type', fontsize=16)  
        plt.ylabel('Weather Variable', fontsize=16)  
        plt.tick_params(axis='both', which='major', labelsize=13) 
        plt.tight_layout()
        plt.show()

    return correlation_df

def analyze_all_conditions(combined_df, lag=0):
    seasons = [('Winter', '12-01', '02-28'), ('Spring', '03-01', '05-31'),
               ('Summer', '06-01', '08-31'), ('Fall', '09-01', '11-30')]
    hour_segments = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
    # hour_segments = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 20), (20, 22), (22, 24)]

    day_types = [('Weekday', False), ('Weekend', True)]

    all_results = []

    for season, start_date, end_date in seasons:
        for start_hour, end_hour in hour_segments:
            for day_type, is_weekend in day_types:
                result_df = analyze_weather_impact(
                    combined_df,
                    lag=lag, start_date=start_date, end_date=end_date,
                    hour_range=(start_hour, end_hour), is_weekend=is_weekend,
                    plot=False
                )
                
                result_df['Season'] = season
                result_df['Hour Segment'] = f'{start_hour:02d}-{end_hour:02d}'
                result_df['Season_Hour'] = f'{season}_{start_hour:02d}-{end_hour:02d}'
                result_df['Day Type'] = day_type
                
                all_results.append(result_df)

    return pd.concat(all_results, ignore_index=True)
