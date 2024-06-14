# Active: A Heart Disease Prediction Application

## Overview
**Active** is a flask-based web application for predicting heart disease using a machine learning model that's trained on the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). The dataset contains 11 features used to predict the presence of heart disease. The application allows users to upload health data, run the trained model to make predictions, and visualize important features influencing the predictions.

## Table of Contents
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Training the Model](#training-the-model)
- [Using the Flask Application](#using-the-flask-application)
- [Model Information](#model-information)
- [Feature Importance](#feature-importance)
- [Application Features](#application-features)



## Project Structure
```bash
.
├── api
│   ├── dataset
│   │   └── heart.csv
│   ├── __init__.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── app
│   ├── models
│   │   └── user.py
│   ├── services
│   │   └── db_ops.py
│   ├── static
│   │   └── css
│   │       └── style.css
│   ├── templates
│   │   ├── base.html
│   │   └── ...
│   ├── views
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── ...
├── config
│   ├── __init__.py
│   ├── config.py
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
```

## Setup and Installation
1. Clone the repository
```bash
git clone https://github.com/<username>/active.git
cd active
```

2. Activate a virtual environment and install the required packages
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Training the Model

### Data
The model is trained on the Heart Failure Prediction Dataset, which contains 918 samples with 11 features each. The dataset includes the following features:

```text
Age
Sex
ChestPainType
RestingBP
Cholesterol
FastingBS
RestingECG
MaxHR
ExerciseAngina
Oldpeak
ST_Slope
```

### Training
To train the model with the best combination of hyperparameters:

1. Navigate to the api directory:
```bash
cd api
```

2. Run the training script with hyperparameter tuning:
```bash
python train.py train xgboost --scale --cv 5
```


## Model Information
The XGBoost model was trained with the following best hyperparameters:

```python
{
    'colsample_bytree': 1.0,
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 100,
    'subsample': 0.9
}
```

The model was trained and validated on 80% of the data and tested on the remaining 20%.

```python
Validation Accuracy: 0.8913
```


## Feature Importance
The top features influencing the predictions are:

```text
ST_Slope_Up
ChestPainType_ASY
Age
Oldpeak
Sex_F
```


## Using the Flask Application
From the root directory, run the following command to start the Flask application:

```python
python run.py
```

Open a web browser and go to http://127.0.0.1:5000 to use the application.


## Application Features
- **Upload Data:** Upload your health data for prediction.
- **Run Model:** Run the trained model to get predictions on the uploaded data.
- **Visualize Features**: View important features influencing the predictions.


## Additional Details

1. Plotting Feature Importance
The feature importance can be visualized using the SHAP library. The plot_feature_importance function from the utils module plots the importance of each feature based on the trained model.

2. Hyperparameter Tuning
The hyperparameters were fine-tuned using GridSearchCV to find the optimal combination for the XGBoost model. The tuning process considered parameters such as n_estimators, learning_rate, max_depth, subsample, and colsample_bytree.

3. CLI Commands
The application includes CLI commands to facilitate training and evaluation of models:

```bash
Train Model: python train.py train [model_name] --scale --cv [number_of_folds]
```
