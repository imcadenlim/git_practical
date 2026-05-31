from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import config


def train_logistic_regression(X_train_scaled, y_train):
    """
    Train Logistic Regression model.
    """

    model = LogisticRegression(
        max_iter=config.LOGISTIC_MAX_ITER
    )

    model.fit(X_train_scaled, y_train)

    return model


def train_random_forest(X_train, y_train):
    """
    Train Random Forest model.
    """

    model = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        random_state=config.RF_RANDOM_STATE
    )

    model.fit(X_train, y_train)

    return model


def train_xgboost(X_train, y_train):
    """
    Train XGBoost model.
    """

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)

    model = XGBClassifier(
        n_estimators=config.XGB_N_ESTIMATORS,
        max_depth=config.XGB_MAX_DEPTH,
        learning_rate=config.XGB_LEARNING_RATE,
        random_state=config.XGB_RANDOM_STATE
    )

    model.fit(X_train, y_train_encoded)

    return model, label_encoder
