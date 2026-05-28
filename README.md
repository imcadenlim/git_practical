# Gas Monitoring Activity Classification

## Project Overview

This project analyzes environmental sensor data collected from elderly residents’ homes to predict resident activity levels using machine learning techniques.

The project includes:

* Exploratory Data Analysis (EDA)
* Data cleaning and preprocessing
* Feature engineering
* Machine learning model comparison
* Model evaluation and interpretation

---

## Machine Learning Models

The following machine learning models were implemented and compared:

1. Logistic Regression
2. Random Forest
3. XGBoost

Random Forest achieved the strongest overall performance on the dataset.

---

## Dataset

The dataset is stored in SQLite database format:

```text
data/gas_monitoring.db
```

---

## Project Structure

```text
EGT309PROJECT/
│
├── data/
│   └── gas_monitoring.db
│
├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── pipeline.py
│
├── saved_model/
├── eda2.ipynb
├── requirements.txt
└── README.md
```

---

## Installation

Install required Python packages:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Navigate to the `src` folder and run:

```bash
python pipeline.py
```

---

## Key Findings

* Missing values and inconsistent labels were identified and corrected during preprocessing.
* Significant class imbalance was observed in the target variable.
* Tree-based ensemble methods outperformed Logistic Regression.
* Metal Oxide Sensors and electrochemical CO2 sensors were the most important predictive features.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* SQLite
* Google Colab
* VS Code
* GitHub
