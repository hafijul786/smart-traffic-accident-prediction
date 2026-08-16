https://smart-traffic-accident-prediction-dfrupcxf35jrcc7srak8an.streamlit.app/

# 🚦 Smart Traffic Accident Severity Prediction

An AI-powered machine learning system that predicts the severity of road accidents based on driver, vehicle, road, weather, traffic, and accident-related conditions.

## 📌 Project Overview

This project uses Machine Learning to predict accident severity into three categories:

- 🟢 Slight Injury
- 🟠 Serious Injury
- 🔴 Fatal Injury

The project also includes an interactive Streamlit web application where users can enter accident-related information and get a predicted severity.

## ✨ Features

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Missing value analysis
- Categorical feature encoding
- Feature engineering
- Train/Test split
- Multiple Machine Learning models
- Model evaluation using accuracy and classification report
- Interactive Streamlit web application
- Real-time accident severity prediction

## 📊 Dataset

The dataset contains:

- **12,316 accident records**
- **32 original features**
- Driver information
- Vehicle information
- Road conditions
- Weather conditions
- Light conditions
- Accident causes
- Accident severity

### Target Variable

`Accident_severity`

Classes:

- Slight Injury
- Serious Injury
- Fatal injury

## 🤖 Machine Learning Models

The project experimented with multiple classification algorithms:

- Logistic Regression
- Random Forest
- Gradient Boosting
- SMOTE-based Logistic Regression

Among the tested models, Gradient Boosting achieved approximately **85% test accuracy**.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
EDA
   ↓
Feature Engineering
   ↓
Categorical Encoding
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Model Saving
   ↓
Streamlit Application