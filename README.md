# 🏭 Hybrid Anomaly Detection in Industrial IoT using Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 1. Title
**Hybrid Anomaly Detection for Predictive Maintenance in Turbofan Engines.**

## ⚠️ 2. Problem Statement
In industrial IoT, equipment failure leads to costly unplanned downtime. The objective of this project is to predict impending equipment failure before it happens by analyzing multi-variate time-series sensor data (e.g., temperature, pressure, vibration). By identifying anomalies in the sensor data early, maintenance can be scheduled proactively, saving time and resources.

## 🧠 3. Methodology
This project applies Deep Learning anomaly detection techniques to time-series telemetry data. 

The solution relies on training an **LSTM Autoencoder** exclusively on "healthy" engine data to establish a baseline of normal operation. The Long Short-Term Memory (LSTM) network is specifically used to capture the complex temporal dependencies and sequential patterns in the sensor data. 

When the trained model is fed data from a degrading engine, it fails to reconstruct the new, erratic patterns. This causes the Reconstruction Error (Mean Squared Error) to spike, effectively acting as an early warning system for total system failure.

## 📊 4. Results
The model successfully identifies engine degradation cycles before catastrophic failure. The graph below plots the Mean Squared Error (MSE) against the sequence time. When the blue line crosses the red threshold, an anomaly is flagged.

![NASA Engine Anomaly Detection Graph](assets/results.png)

## 💾 5. Dataset
The data used is the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) Dataset, containing simulated run-to-failure telemetry from turbofan engines.
* **Dataset Source:** [Kaggle: CMAPSS Jet Engine Simulated Data](https://www.kaggle.com/datasets/palbha/cmapss-jet-engine-simulated-data)
* *Note: The raw data files are not included in this repository. Please download them from the link above and place them in the `data/` directory.*

## 🚀 6. How to Run

**1. Clone the repository:**
`git clone https://github.com/Atharva-Ramawat/Industrial-IOT-Anomaly-Detection.git`
`cd Industrial-IOT-Anomaly-Detection`

**2. Set up the Python 3.11 Virtual Environment:**
`py -3.11 -m venv venv`
`.\venv\Scripts\activate`
`pip install -r requirements.txt`

**3. Train the Model:**
*(Ensure your NASA data text files are placed in the `data/` folder first)*
`python src/train.py`

**4. Run Predictions & Visualize Anomalies:**
`python src/predict.py`

## 🔮 6. Future Work & Next Steps
While the current LSTM Autoencoder successfully identifies engine degradation, this project can be expanded into a full-scale production system through the following enhancements:

* **Real-Time IoT Simulation:** Implement Apache Kafka or a continuous Python generator to stream the test dataset sequentially, simulating a live factory floor environment.
* **API Deployment:** Wrap the inference script (`predict.py`) inside a **FastAPI** application to allow external systems to send batches of sensor data and receive JSON responses flagging potential anomalies.
* **Remaining Useful Life (RUL) Prediction:** Transition from a purely unsupervised anomaly detection approach to a supervised regression model that calculates exactly *how many cycles* remain before engine failure using the `RUL_FD001.txt` ground truth data.
* **Interactive Dashboarding:** Build a **Streamlit** frontend to visualize the live streaming sensor data and dynamic reconstruction error graphs for non-technical stakeholders.
