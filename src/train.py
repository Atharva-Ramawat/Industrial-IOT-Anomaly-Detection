import os
import numpy as np
# Import your custom functions from the other files
from data_loader import load_data, drop_flatline_sensors, scale_data, generate_sequences
from model import build_lstm_autoencoder

# --- Configuration ---
SEQUENCE_LENGTH = 50
BATCH_SIZE = 64
EPOCHS = 20

def main():
    print("🚀 Starting Training Pipeline...")
    
    # 1. Load Data
    train_df = load_data('data/train_FD001.txt')
    test_df = load_data('data/test_FD001.txt')
    
    # 2. Clean Data (Drop flatline sensors)
    # Note: Make sure your data_loader.py returns the cleaned dataframes!
    train_df, test_df = drop_flatline_sensors(train_df, test_df)
    
    # Define which columns are actual sensor features (ignoring unit_number, time_cycles, etc.)
    # Adjust this list based on what is left after dropping flatlines
    feature_cols = [col for col in train_df.columns if 'sensor' in col or 'setting' in col]
    num_features = len(feature_cols)
    
    # 3. Scale Data
    train_df, test_df, scaler = scale_data(train_df, test_df, feature_cols)
    
    # 4. Generate 3D Sequences for LSTM
    print("Creating sequences...")
    # Autoencoders use the same data for input (X) and target (y)
    X_train = generate_sequences(train_df, SEQUENCE_LENGTH, feature_cols)
    
    # Convert lists to numpy arrays for TensorFlow
    X_train = np.array(X_train)
    print(f"Final Training Shape: {X_train.shape}")
    
    # 5. Build and Train the Model
    print("Building model...")
    model = build_lstm_autoencoder(SEQUENCE_LENGTH, num_features)
    
    print("Training model...")
    history = model.fit(
        X_train, X_train, # X and y are the same for autoencoders
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1, # Use 10% of training data to check for overfitting
        verbose=1
    )
    
    # 6. Save the trained model
    os.makedirs('../models', exist_ok=True)
    model_path = '../models/lstm_autoencoder.keras'
    model.save(model_path)
    print(f"✅ Model successfully saved to {model_path}")

if __name__ == "__main__":
    # Ensure we are running from the src directory context
    main()