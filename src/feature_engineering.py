import pandas as pd

def add_features(df):
    # Example: interaction features or derived columns
    if 'num_lab_procedures' in df.columns and 'num_medications' in df.columns:
        df['labs_per_med'] = df['num_lab_procedures'] / (df['num_medications'] + 1)
    return df
