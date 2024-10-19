import numpy as np

# Function to calculate the log growth rate for any 6-month period
def calculate_log_monthly_growth(df):

    # Sort data to ensure chronological order
    df = df.sort_values(by=['year', 'month'])
    
    # Calculate month-to-month log growth rate
    df['log_growth_rate'] = (np.log(df['total_contributions']) - np.log(df['total_contributions'].shift(1))) * 100

    # Drop the first month, as its growth rate will be NaN
    df = df.dropna(subset=['log_growth_rate'])
    
    # Create a sequential 'x-axis' column (1, 2, 3, ..., months_to_consider)
    df['x_axis'] = np.arange(1, 7)

    return df