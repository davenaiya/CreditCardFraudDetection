# Credit Card Fraud Detection

A machine learning project that explores credit card transactions and predicts whether a generated transaction is likely fraudulent. It includes a Jupyter notebook for data analysis and model experiments, plus a simple Streamlit frontend.

## Overview

The labeled transaction dataset is used to train the model. In the frontend, click **Generate transaction and predict** to create a synthetic transaction and score it. The app displays the predicted result, fraud and not-fraud probabilities, and the feature values used for that prediction.

Predictions are model estimates for demonstration. They are not confirmed fraud findings or real payment decisions.

## Features

- Explore the dataset and model experiments in the notebook.
- Train from labeled transaction data (`Class`: `0` = normal, `1` = fraud).
- Generate a synthetic transaction and predict its class in the frontend.
- View the prediction probability graph and generated feature values.
- Use notebook model artifacts when present, or train the app's balanced Random Forest on startup.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- Jupyter Notebook

## Screenshots

### Fraud prediction

![Transaction Fraud Check frontend](screenshots/fraud-prediction.png)

## Project Structure

```text
CreditCardFraudDetection/
├── app.py                    # Streamlit frontend and prediction flow
├── creditcard_project.ipynb  # Data analysis and model experiments
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── screenshots/
│   └── fraud-prediction.png  # Streamlit frontend screenshot
└── .gitignore                # Excludes local data and generated files
```

