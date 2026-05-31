# Gas Monitoring Activity Classification

## Project Overview

This project analyzes environmental sensor data collected from elderly residents’ homes to predict resident activity levels using machine learning techniques.

The project follows a complete machine learning workflow:

1. Data ingestion from SQLite database
2. Exploratory Data Analysis (EDA)
3. Data cleaning and preprocessing
4. Feature engineering
5. Machine learning model training
6. Model evaluation and comparison
7. Feature importance analysis

The objective is to determine whether environmental sensor measurements can be used to accurately classify resident activity levels.

---

## Dataset

The dataset is stored in SQLite database format:

```text
data/gas_monitoring.db
```

The dataset contains environmental measurements including:

* Temperature
* Humidity
* CO2 sensor readings
* Metal Oxide gas sensor readings
* HVAC operation modes
* Ambient light levels

The target variable is:

```text
Activity Level
```

which contains activity categories such as:

* Low Activity
* Moderate Activity
* High Activity

---

## Key EDA Findings

Several important observations were identified during exploratory data analysis:

### Missing Values

Missing values were identified in multiple sensor variables, including:

* Humidity
* MetalOxideSensor_Unit2
* CO_GasSensor
* Ambient Light Level

Median imputation was selected because it is robust to extreme values and preserves the overall distribution of sensor measurements.

### Data Quality Issues

The dataset contained inconsistent activity labels such as:

```text
Moderate Activity
ModerateActivity
```

These labels were standardized during preprocessing to ensure consistent model training.

### Class Imbalance

The dataset exhibited class imbalance, with Low Activity representing the majority class.

This observation motivated the use of multiple evaluation metrics beyond simple accuracy.

### Correlation Analysis

Most sensor variables showed weak to moderate correlations, suggesting that multiple environmental measurements contribute unique information to the prediction task.

---

## Feature Engineering

The following preprocessing and feature engineering steps were applied:

* Missing value imputation using median values
* Standardization of inconsistent categorical labels
* One-hot encoding of categorical variables
* Label encoding of the target variable
* Feature scaling using StandardScaler for Logistic Regression

These steps were implemented to improve data quality and model performance.

---

## Machine Learning Models

Three machine learning models were implemented and compared.

### Logistic Regression

Logistic Regression was selected as a baseline classification model due to its simplicity, interpretability, and computational efficiency.

### Random Forest

Random Forest was selected because it can model non-linear relationships, handle noisy sensor data, and provide feature importance scores.

### XGBoost

XGBoost was selected because of its strong predictive performance and ability to capture complex relationships within structured datasets.

---

## Evaluation Metrics

Multiple evaluation metrics were used:

### Accuracy

Measures overall prediction correctness.

### Precision

Measures how many predicted labels are correct.

### Recall

Measures how many actual labels are correctly identified.

### F1-Score

Provides a balanced measure between precision and recall.

Because the dataset is imbalanced, F1-score, precision, and recall were considered alongside accuracy.

---

## Model Performance Summary

The models were evaluated using classification reports and performance metrics.

Overall findings:

* Random Forest achieved the strongest overall performance.
* Logistic Regression provided a useful baseline model.
* XGBoost produced competitive results but did not outperform Random Forest on this dataset.

---

## Feature Importance

Random Forest feature importance analysis indicated that:

* MetalOxideSensor_Unit4
* MetalOxideSensor_Unit3
* CO2_ElectroChemicalSensor
* MetalOxideSensor_Unit2

were among the most influential predictors of resident activity level.

These findings suggest that gas sensor measurements play an important role in activity prediction.

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

## File Contributions

Project implementation:

| File              | Responsibility                  |
| ----------------- | ------------------------------- |
| data_ingestion.py | Load data from SQLite database  |
| preprocessing.py  | Data cleaning and preprocessing |
| train_model.py    | Model training                  |
| evaluate_model.py | Model evaluation                |
| pipeline.py       | Pipeline orchestration          |
| eda2.ipynb        | Exploratory Data Analysis       |

---

## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Navigate to the source folder:

```bash
cd src
```

Run the machine learning pipeline:

```bash
python pipeline.py
```

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

---

## Conclusion

The project demonstrates that environmental sensor measurements can be used to predict resident activity levels. After comparing multiple machine learning algorithms, Random Forest achieved the strongest overall performance and provided useful insights into the most important environmental predictors of resident activity.
