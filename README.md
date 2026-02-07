# Customer Churn Prediction using ANN and Streamlit

## 📌 Project Description
Customer churn prediction is a machine learning project that predicts whether a customer is likely to leave a company based on historical customer data.  
This project uses an **Artificial Neural Network (ANN)** for prediction and is deployed as a **Streamlit web application** for real-time interaction.

---

## 🎯 Objective
- Predict customer churn using ANN
- Understand customer behavior patterns
- Build an end-to-end ML pipeline
- Deploy the trained model using Streamlit

---

## 🧠 Machine Learning Model
- Model Type: Artificial Neural Network (ANN)
- Problem Type: Binary Classification
- Output:  
  - `1` → Customer will churn  
  - `0` → Customer will not churn

---

## 📊 Dataset
- Contains customer demographic and account-related information
- Includes both numerical and categorical features
- Target variable: **Churn**

---

## 🔄 Data Preprocessing
- Handling categorical variables using encoding
- Feature scaling using StandardScaler / MinMaxScaler
- Splitting data into training and testing sets

---

## 🏗️ ANN Architecture
- Input Layer: Customer features
- Hidden Layers: Multiple layers with ReLU activation
- Output Layer: Sigmoid activation
- Optimizer: Adam
- Loss Function: Binary Crossentropy

---

## 🌐 Web Application (Streamlit)
- User-friendly interface for input
- Loads trained ANN model
- Performs real-time churn prediction
- Displays prediction result instantly

---

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Streamlit

---

## 📂 Project Structure
Customer-Churn-Prediction/
│
├── app.py # Streamlit application
├── model.h5 # Trained ANN model
├── scaler.pkl # Saved scaler
├── Churn_Modelling.csv # Dataset
├── requirements.txt # Required libraries
└── README.md # Project documentat+
