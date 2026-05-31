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

## Team Information

**Project Type:** Individual Project

**Student:**
- Caden Lim

All project components, including the EDA notebook, machine learning pipeline, documentation, and GitHub repository management, were completed independently.

---

## Dataset

The dataset is stored in SQLite database format:

```text
data/gas_monitoring.db
```

The dataset contains environmental measurements including:

- Temperature
- Humidity
- CO2 sensor readings
- Metal Oxide gas sensor readings
- HVAC operation modes
- Ambient light levels

The target variable is:

```text
Activity Level
```

which contains activity categories such as:

- Low Activity
- Moderate Activity
- High Activity

---

## Key EDA Findings

Several important observations were identified during exploratory data analysis.

### Missing Values

Missing values were identified in multiple sensor variables, including:

- Humidity
- MetalOxideSensor_Unit2
- CO_GasSensor
- Ambient Light Level

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

- Missing value imputation using median values
- Standardization of inconsistent activity labels
- Standardization of HVAC operation mode labels
- One-hot encoding of categorical variables
- Label encoding for XGBoost
- Feature scaling using StandardScaler for Logistic Regression

These transformations improve data quality and ensure compatibility with machine learning algorithms.

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

## Hyperparameter Tuning

To improve model performance, hyperparameter tuning was performed on the Random Forest model using RandomizedSearchCV.

Parameters such as:

- Number of trees (n_estimators)
- Maximum tree depth (max_depth)
- Minimum samples required for splitting

were evaluated across multiple parameter combinations.

RandomizedSearchCV was selected because it allows efficient exploration of the hyperparameter search space while requiring less computational time than an exhaustive Grid Search.

The tuned Random Forest model was then compared against the baseline models to determine whether performance improvements could be achieved.

---

## Evaluation Metrics

Multiple evaluation metrics were used to evaluate model performance.

### Accuracy

Measures the overall proportion of correct predictions.

### Precision

Measures how many predicted labels are correct.

### Recall

Measures how many actual labels are correctly identified.

### F1-Score

Provides a balanced measure between precision and recall.

### Confusion Matrix

Provides detailed insight into classification performance for each activity class.

Because the dataset is imbalanced, precision, recall, and F1-score were considered alongside accuracy.

---

## Model Performance Summary

The models were evaluated using classification reports and confusion matrices.

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 0.63 |
| Random Forest | 0.69 |
| XGBoost | 0.65 |

Random Forest achieved the strongest overall performance and was selected as the best-performing model.

---

## Feature Importance

Random Forest feature importance analysis indicated that:

- MetalOxideSensor_Unit4
- MetalOxideSensor_Unit3
- CO2_ElectroChemicalSensor
- MetalOxideSensor_Unit2

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
├── eda.ipynb
├── requirements.txt
├── run.sh
└── README.md
```

---

## Contribution Summary

This project was completed individually.

All files were authored and maintained by Rahmat.

| File | Responsibility |
|--------|--------|
| data_ingestion.py | Load data from SQLite database |
| preprocessing.py | Data cleaning and preprocessing |
| train_model.py | Machine learning model training |
| evaluate_model.py | Model evaluation and reporting |
| pipeline.py | End-to-end pipeline orchestration |
| eda.ipynb | Exploratory Data Analysis and findings |
| README.md | Project documentation |

---

## Installation

Install the required packages:

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

Alternatively, from the project root:

```bash
sh run.sh
```

---

## Docker

Docker was not used in this project.

The machine learning pipeline can be executed directly using Python using the instructions provided above.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- SQLite
- Jupyter Notebook
- Google Colab
- VS Code
- GitHub

---

## Conclusion

This project demonstrated that environmental sensor measurements can be used to predict resident activity levels using machine learning techniques.

Several data quality issues, including missing values and inconsistent labels, were identified and corrected during preprocessing. Multiple machine learning models were evaluated, with Random Forest achieving the strongest overall performance.

Feature importance analysis revealed that gas sensor measurements were among the most influential predictors of activity level, highlighting the value of environmental sensing data for activity monitoring applications.

Future improvements could include additional feature engineering, advanced class balancing techniques such as SMOTE, and further hyperparameter optimization to improve minority-class prediction performance.