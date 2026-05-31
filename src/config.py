# config.py
# Central configuration file for the ML pipeline.
# Modify values here to change pipeline behaviour without editing source code.

import os

# --- Database ---
# Supports both local (../data/) and Docker (/app/data/) environments
DATABASE_PATH = os.environ.get(
    "DATABASE_PATH",
    "../data/gas_monitoring.db"
)

# --- Train/Test Split ---
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Model Hyperparameters ---

# Logistic Regression
LOGISTIC_MAX_ITER = 1000

# Random Forest
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42

# XGBoost
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.1
XGB_RANDOM_STATE = 42

# --- Saved Model Output ---
SAVED_MODEL_DIR = os.environ.get(
    "SAVED_MODEL_DIR",
    "../saved_model"
)
