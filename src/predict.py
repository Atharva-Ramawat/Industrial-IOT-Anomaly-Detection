import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import load_data, drop_flatline_sensors, scale_data, generate_sequences

# Configuration
SEQUENCE_LENGTH = 50
# This threshold is the "red line". Anything above this MSE is considered an anomaly.
# You might need to tweak this number up or down based on your specific results!
THRESHOLD = 0.015 

def main():
    print("🔍 Starting Anomaly Detection Pipeline...")
    
    # 1. Load and prep the data 
    # (We load train_df just so the scaler fits exactly the same way it did during training)
    train_df = load_data('data/train_FD001.txt')
    test_df = load_data('data/test_FD001.txt')
    
    train_df, test_df = drop_flatline_sensors(train_df, test_df)
    feature_cols = [col for col in train_df.columns if 'sensor' in col or 'setting' in col]
    
    train_df, test_df, scaler = scale_data(train_df, test_df, feature_cols)
    
    # 2. Generate Sequences for Test Data
    print("Preparing test sequences...")
    X_test = generate_sequences(test_df, SEQUENCE_LENGTH, feature_cols)
    X_test = np.array(X_test)
    
    # 3. Load the Trained Model
    print("Loading trained model...")
    model = tf.keras.models.load_model('models/lstm_autoencoder.keras')
    
    # 4. Make Predictions
    print("Reconstructing test data...")
    reconstructions = model.predict(X_test)
    
    # 5. Calculate Mean Squared Error (MSE)
    # This checks how badly the model struggled to recreate the input.
    mse = np.mean(np.power(X_test - reconstructions, 2), axis=(1, 2))
    
    # 6. Flag Anomalies
    anomalies = mse > THRESHOLD
    print(f"\n🚨 Detected {np.sum(anomalies)} anomalies out of {len(mse)} total sequences.")
    
    # 7. Visualize the Results
    print("Plotting results...")
    plt.figure(figsize=(12, 6))
    plt.plot(mse, label='Reconstruction Error (MSE)', color='blue', alpha=0.6)
    plt.axhline(y=THRESHOLD, color='red', linestyle='--', label=f'Threshold ({THRESHOLD})')
    
    # Highlight the actual anomalies in red dots
    anomaly_indices = np.where(anomalies)[0]
    plt.scatter(anomaly_indices, mse[anomaly_indices], color='red', s=10, label='Anomaly Detected')
    
    plt.title('NASA Engine Anomaly Detection')
    plt.xlabel('Test Sequence Index (Time)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()