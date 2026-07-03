# Credit Card Fraud Detection

A machine learning project that detects fraudulent bank transactions 
and deploys the model as an interactive web app using Streamlit.

---

## About the Project

I built this project to learn how machine learning can be used to 
solve real-world financial fraud problems. The model is trained on 
6.3 million mobile money transactions and predicts whether a 
transaction is fraudulent or not.

---

## Dataset

- **Source:** [PaySim Synthetic Financial Dataset - Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Size:** 6.3 million transactions
- **Fraud rate:** Only 0.13% of transactions are fraud
- **Transaction types:** PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT

---

## What I Did

- Explored and visualized the dataset (fraud rates, transaction types, amount distribution)
- Found that only TRANSFER and CASH_OUT type transactions had fraud
- Handled severe class imbalance using class_weight='balanced'
- Built a preprocessing pipeline with StandardScaler and OneHotEncoder
- Trained a Logistic Regression model
- Evaluated using classification report and confusion matrix
- Saved the model and deployed it as a Streamlit web app

---

## Model Comparison

| Model | Accuracy | Fraud Recall | Fraud Precision | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 94.70% | 92.53% | 2.21% | 4.31% |
| Decision Tree | 99.46% | 98.54% | 19.18% | 32.11% |
| **Random Forest** | **99.97%** | **76.79%** | **97.08%** | **85.75%** |

Random Forest selected as final model — highest accuracy and precision.
---

## Tech Used

- Python 3.14
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- Joblib

---

## How to Run Locally

1. Clone the repo
git clone https://github.com/Mayankr-1508/fraud-detection-ml.git

2. Install libraries
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib

3. Download the dataset from Kaggle link above and place it in the folder

4. Run the notebook first to generate the model
Open anaysis_model.ipynb and run all cells

5. Launch the web app
streamlit run fraud_detection.py

---

## Project Structure

fraud-detection-ml/
├── anaysis_model.ipynb        → Data analysis + model training
├── fraud_detection.py         → Streamlit web app
├── fraud_detection_pipeline.pkl  → Saved trained model
└── README.md

---

## App Preview

Enter transaction details like type, amount and balances.
Click Predict to instantly see if the transaction is fraud or not.

---

Built by Mayank Raj
