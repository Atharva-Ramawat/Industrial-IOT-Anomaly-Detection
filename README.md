# Hybrid Anomaly Detection in Industrial IoT using Deep Learning

**Video Explanation:** [Link to your 8-10 Min Video will go here]

## 1. Title
Hybrid Anomaly Detection for Predictive Maintenance in Turbofan Engines.

## 2. Problem Statement
In industrial IoT, equipment failure leads to costly unplanned downtime. The objective of this project is to predict impending equipment failure before it happens by analyzing multi-variate time-series sensor data (e.g., temperature, pressure, vibration). By identifying anomalies in the sensor data early, maintenance can be scheduled proactively.

## 3. Explanation (Methodology)
This project explores, analyzes, and applies Deep Learning anomaly detection techniques to time-series telemetry data. The solution relies on training models exclusively on "healthy" engine data to establish a baseline of normal operation. When the model is fed data from a degrading engine, its reconstruction error spikes, signaling an anomaly. 

To evaluate performance, the project compares two distinct Deep Learning architectures:
1. **LSTM Autoencoder:** Utilizes Long Short-Term Memory networks to capture the temporal dependencies and sequential patterns in the sensor data.
2. **1D-CNN Autoencoder:** Utilizes 1-Dimensional Convolutional Neural Networks to extract spatial-temporal features and sudden local spikes across the sensor readings.

Both models are evaluated based on their Mean Absolute Error (MAE) for signal reconstruction and their ability to detect anomalies prior to total system failure.

## 4. Dataset Link
The data used is the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) Dataset, containing simulated run-to-failure telemetry from turbofan engines.
* **Dataset Source:** [Kaggle: CMAPSS Jet Engine Simulated Data](https://www.kaggle.com/datasets/palbha/cmapss-jet-engine-simulated-data)
* *Note: The raw data files are not included in this repository due to size constraints. Please download them from the link above and place them in the `/data` directory to run the notebook.*

## 5. How to Run
1. Clone this repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Download the dataset from the link above and place the `train_FD001.txt` and `test_FD001.txt` files inside the `data/` folder.
4. Open and run `notebooks/anomaly_detection_model.ipynb`.
