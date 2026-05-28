from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


def train_logistic_regression(X_train_scaled, y_train):
    """
    Train Logistic Regression model.
    """

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_scaled, y_train)

    return model


def train_random_forest(X_train, y_train):
    """
    Train Random Forest model.
    """

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
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
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train_encoded)

    return model, label_encoder