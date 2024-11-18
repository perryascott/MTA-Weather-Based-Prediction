import pandas as pd

class TimeInterval():

    def __init__(self, season, day_type, hour_range):
        self.season = season
        self.day_type = day_type
        self.hour_range = hour_range

    def filter(self, df):
        model_data = df.copy()
        
        # Apply hour range filter if specified
        if self.hour_range is not None:
            model_data = filter_df_by_hour_range(model_data, self.hour_range)
        
        if self.season != 'all':
            model_data = model_data[model_data['season'] == self.season]
        if self.day_type != 'all':
            is_weekend = 1 if self.day_type == 'weekend' else 0
            model_data = model_data[model_data['is_weekend'] == is_weekend]
        
        return model_data
    
    @property
    def summary(self):
        return f"{self.season} {self.day_type} {self.hour_range}"

def filter_df_by_hour_range(df, hour_range):
    if isinstance(hour_range, int):
        df = df[df['hour'].dt.hour == hour_range]
    elif isinstance(hour_range, tuple) and len(hour_range) == 2:
        df = df[(df['hour'].dt.hour >= hour_range[0]) & (df['hour'].dt.hour < hour_range[1])]
    return df

def filter_date_range(data, season='all', day_type='all', hour_range=None):
    """
    Filter data based on the given conditions.
    """
    model_data = data.copy()
    
    # Apply hour range filter if specified
    if hour_range is not None:
        model_data = filter_df_by_hour_range(model_data, hour_range)
    
    if season != 'all':
        model_data = model_data[model_data['season'] == season]
    if day_type != 'all':
        is_weekend = 1 if day_type == 'weekend' else 0
        model_data = model_data[model_data['is_weekend'] == is_weekend]
    
    return model_data

def split_df_at_datetime(df, split_datetime, is_weather=False):
    """
    Split a dataframe into two parts at the specified datetime
    
    Args:
        df: pandas DataFrame containing datetime column
        split_datetime: datetime object to split on
        is_weather: bool indicating if this is weather data (True) or ridership data (False)
    
    Returns:
        tuple of (before_df, after_df)
    """
    # Get the datetime column name based on data type
    dt_col = "time" if is_weather else "hour"
    
    # Convert column to datetime if not already
    df[dt_col] = pd.to_datetime(df[dt_col])
    
    # Split into before and after
    before_df = df[df[dt_col] < split_datetime].copy()
    after_df = df[df[dt_col] >= split_datetime].copy()
    
    return before_df, after_df

