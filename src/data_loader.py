import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Define the exact column names based on NASA's documentation
COLUMNS = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
          [f'sensor_{i}' for i in range(1, 22)]

def load_data(file_path):
    """Reads the raw .txt file and returns a Pandas DataFrame."""
    df = pd.read_csv(file_path, sep='\s+', header=None, names=COLUMNS)
    return df

def drop_flatline_sensors(train_df, test_df):
    """Removes sensors that provide no useful information (variance = 0)."""
    # Find sensors that have exactly 1 unique value across the whole dataset
    nunique = train_df.nunique()
    drop_cols = nunique[nunique == 1].index.tolist()
    
    print(f"Dropping flatline sensors: {drop_cols}")
    
    # Drop them from both datasets to keep features consistent
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)
    
    # EXACTLY TWO items returned here!
    return train_df, test_df

def scale_data(train_df, test_df, feature_cols):
    """Scales data between 0 and 1 so neural networks train faster."""
    scaler = MinMaxScaler()
    
    # IMPORTANT: We only 'fit' the scaler on training data to prevent data leakage!
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    return train_df, test_df, scaler

def generate_sequences(df, sequence_length, feature_cols):
    """
    Converts 2D tabular data into 3D overlapping sequences.
    Required shape for Deep Learning: (Number_of_Samples, Time_Steps, Features)
    """
    sequences = []
    
    # We must generate sequences PER engine (unit_number), not across the whole dataset
    for unit_id in df['unit_number'].unique():
        # Get all rows for this specific engine
        engine_data = df[df['unit_number'] == unit_id][feature_cols].values
        
        # Slide a window of 'sequence_length' over the engine's data
        for i in range(len(engine_data) - sequence_length + 1):
            sequences.append(engine_data[i : i + sequence_length])
            
    return np.array(sequences)

# --- Quick Test Block ---
# If you run this script directly, it will test if your functions work
if __name__ == "__main__":
    print("Testing Data Loader module...")
    try:
        # Assuming we run this from the root folder
        train = load_data('data/train_FD001.txt')
        test = load_data('data/test_FD001.txt')
        print(f"Successfully loaded Train shape: {train.shape}")
        print(f"Successfully loaded Test shape: {test.shape}")
    except FileNotFoundError:
        print("Error: Could not find the data files. Make sure you run this from the root directory.")