# Credit Card Fraud Detection using Machine Learning

## Project Overview

This project builds an end-to-end machine learning pipeline capable of detecting fraudulent credit card transactions.

The pipeline begins by combining multiple raw datasets before cleaning, transforming and engineering new features. Multiple machine learning classification models are then trained, evaluated and compared to determine which performs best at identifying fraudulent transactions.

The project demonstrates a complete machine learning workflow including:

- Extract, Transform and Load (ETL)
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Perparation
- Model Training
- Model Evaluation
- Hyperparameter Tuning
- Model Selection

---

## Problem 

Credit card fraud costs financial institutions billions of pounds every year. Traditional rule-based systems often struggle to detect new fraud patterns while keeping false alarms low.

The objective of this project is to build a machine learning model capable of automatically distinguishing fraudulent transactions from legitimate ones using historical transaction data.

---

## Project Structure

```

credit-card-fraud-detection/

│

├── data/

│ ├── raw/

│ └── processed/

│

├── docs/

| └── final_report.md

|

├── notebooks/

│ └── credit-card-fraud-detection.ipynb

|

├── outputs/

│ ├── evaluation/

│ ├── models/

│ └── results/

│

├── src/

│ ├── clean_data.py

│ ├── features.py

│ ├── preparation.py

│ ├── train_models.py

│ ├── evaluate.py

│ └── tune_best_model.py

│

├── README.md

├── requirements.txt

└── .gitignore

```

---

## Machine Learning Pipeline

### 1. `clean_data.py`

- Loads all three datasets
- Merges customer and terminal information
- Cleans missing values
- Removes duplicate records
- Saves a cleaned dataset

---

### 2. `features.py`

Additional behavioural features are created including:

- Hour of transaction
- Day of week
- Weekend indicator
- Night transaction
- Amount difference from customer average
- Amount ratio
- Transaction z-score
- Minutes since previous transaction
- Terminal changed
- Rapid terminal change
- Distance between customer and terminal

These engineered features help the models identify suspicious customer behaviour rather than relying only on the original transaction data.

---

### 3. `preparation.py`

- Drops unnecessary columns
- Splits features and labels
- Train/Test split
- Standardises numerical variables
- One-hot encodes categorical variables

A Scikit-Learn `ColumnTransformer` is used so preprocessing is automatically applied during model training.

---

### 4. `train_models.py`

Three classification models are trained.

- Logistic Regression
- Random Forest
- Gradient Boosting

Each model is saved as a `.pkl` file for later evaluation.

---

### 5. `evaluate.py`

Each saved model is evaluated using:

- Classification Report
- Confusion Matrix
- ROC Curve
- ROC-AUC Score

The script automatically identifies the best performing model and produces evaluation figures.

---

### 6. `tune_best_model.py`

The recommended model is tuned using GridSearchCV.

The tuned model is saved separately and can then be re-evaluated.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Joblib
- Jupyter Notebook

