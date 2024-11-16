import pandas as pd

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

