# Customer Intelligence & Churn Analytics Platform

## Overview

Customer churn is one of the most important business metrics for subscription-based companies. This project predicts whether a customer is likely to churn based on demographic information, service usage patterns, contract details, and billing information.

The project was built as an end-to-end Machine Learning application covering data ingestion, preprocessing, model training, model evaluation, serialization, inference pipelines, and deployment using Flask.

---

## Problem Statement

Customer acquisition is significantly more expensive than customer retention. The objective of this project is to identify customers who are likely to leave so that businesses can proactively take retention actions.

---

## Dataset

* Records: 100,000+ customers
* Features:

  * Age
  * Gender
  * Tenure
  * Monthly Charges
  * Contract Type
  * Payment Method
  * Total Charges
* Target:

  * Churn (Yes / No)

---

## Tech Stack

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier

### Deployment

* Flask

### Version Control

* Git
* GitHub

---

## Project Architecture

data/
↓
Data Ingestion
↓
Data Transformation
↓
Model Training
↓
Model Selection
↓
Model Serialization
↓
Prediction Pipeline
↓
Flask Application

---

## Machine Learning Workflow

### Data Ingestion

* Dataset loading
* Train/Test split
* Stratified sampling

### Data Transformation

* OneHot Encoding
* Standard Scaling
* ColumnTransformer
* Pipeline-based preprocessing

### Model Training

Models evaluated:

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score

---

## Model Performance

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Logistic Regression | 72.35%   | 0.5277   |
| Random Forest       | 72.86%   | 0.5531   |
| Gradient Boosting   | 76.23%   | 0.6094   |

### Best Model

Gradient Boosting Classifier

---

## Features

* End-to-End ML Pipeline
* Modular Project Structure
* Reusable Data Transformation Pipeline
* Model Serialization using Joblib
* Real-Time Churn Prediction
* Flask-Based User Interface

---

## Future Improvements

* Hyperparameter Tuning
* Cross Validation
* Feature Importance Analysis
* Docker Containerization
* AWS Deployment
* MLflow Experiment Tracking

---

## Author

Raj Parihar

B.Tech Computer Science Engineering
Aspiring Data Scientist / Machine Learning Engineer
