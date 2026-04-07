import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data_loader import load_data, drop_flatline_sensors, scale_data, generate_sequences

# --- Configuration ---
SEQUENCE_LENGTH = 50
THRESHOLD = 0.015 
FAILURE_WINDOW = 30 # A machine is considered in "Anomaly/Failure" state in its final 30 cycles

def main():
    print("🔍 Starting Anomaly Detection & Evaluation Pipeline...")
    
    # ==========================================
    # 1. LOAD AND PREP SENSOR DATA
    # ==========================================
    # Load train_df just to fit the scaler accurately
    train_df = load_data('data/train_FD001.txt')
    raw_test_df = load_data('data/test_FD001.txt')
    
    train_clean, test_clean = drop_flatline_sensors(train_df, raw_test_df.copy())
    feature_cols = [col for col in train_clean.columns if 'sensor' in col or 'setting' in col]
    
    train_scaled, test_scaled, scaler = scale_data(train_clean, test_clean, feature_cols)
    
    print("Preparing test sequences...")
    X_test = generate_sequences(test_scaled, SEQUENCE_LENGTH, feature_cols)
    X_test = np.array(X_test)
    
    # ==========================================
    # 2. RUN MODEL PREDICTIONS
    # ==========================================
    print("Loading trained model and reconstructing test data...")
    model = tf.keras.models.load_model('models/lstm_autoencoder.keras')
    reconstructions = model.predict(X_test)
    
    # Calculate MSE (Error Score)
    mse = np.mean(np.power(X_test - reconstructions, 2), axis=(1, 2))
    
    # Flag Anomalies based on your threshold
    anomalies = mse > THRESHOLD
    y_pred = anomalies.astype(int) # Convert True/False to 1/0
    
    # ==========================================
    # 3. CALCULATE GROUND TRUTH (y_true)
    # ==========================================
    print("Calculating Ground Truth from RUL data...")
    rul = pd.read_csv('data/RUL_FD001.txt', sep='\s+', header=None, names=['RUL'])
    
    # Calculate actual RUL for every single row
    rul_calc = raw_test_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    rul_calc['RUL_true'] = rul['RUL'].values
    raw_test_df = raw_test_df.merge(rul_calc[['unit_number', 'RUL_true']], on='unit_number')
    
    max_cycles = raw_test_df.groupby('unit_number')['time_in_cycles'].transform('max')
    raw_test_df['Actual_RUL'] = raw_test_df['RUL_true'] + (max_cycles - raw_test_df['time_in_cycles'])
    
    # Label row as 1 (Anomaly) if it's within the final FAILURE_WINDOW cycles
    raw_test_df['Is_Anomaly'] = (raw_test_df['Actual_RUL'] <= FAILURE_WINDOW).astype(int)
    
    # Align labels with the 50-step sequences
    y_true = []
    for engine_id in raw_test_df['unit_number'].unique():
        engine_data = raw_test_df[raw_test_df['unit_number'] == engine_id]
        labels = engine_data['Is_Anomaly'].values
        # Grab the label of the last row in each 50-step sequence
        for i in range(len(labels) - SEQUENCE_LENGTH + 1):
            y_true.append(labels[i + SEQUENCE_LENGTH - 1])
            
    y_true = np.array(y_true)

    # ==========================================
    # 4. PRINT PERFORMANCE METRICS
    # ==========================================
    print("\n" + "="*40)
    print("📊 MODEL PERFORMANCE METRICS")
    print("="*40)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
 
    
    print("\nConfusion Matrix (TN, FP | FN, TP):")
    print(confusion_matrix(y_true, y_pred))
    print("="*40 + "\n")

    # ==========================================
    # 5. PLOT THE GRAPH
    # ==========================================
    print("Plotting anomaly detection graph...")
    plt.figure(figsize=(12, 6))
    plt.plot(mse, label='Reconstruction Error (MSE)', color='blue', alpha=0.6)
    plt.axhline(y=THRESHOLD, color='red', linestyle='--', label=f'Threshold ({THRESHOLD})')
    
    # Highlight anomalies in red dots
    anomaly_indices = np.where(anomalies)[0]
    plt.scatter(anomaly_indices, mse[anomaly_indices], color='red', s=10, label='Anomaly Detected')
    
    plt.title('NASA Engine Anomaly Detection (Test Data)')
    plt.xlabel('Test Sequence Index (Time)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()