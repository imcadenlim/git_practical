import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def clean_activity_labels(dataframe):
    """
    Standardize inconsistent activity labels.
    """

    dataframe["Activity Level"] = dataframe["Activity Level"].replace({
        "LowActivity": "Low Activity",
        "Low_Activity": "Low Activity",
        "ModerateActivity": "Moderate Activity"
    })

    return dataframe


def clean_hvac_labels(dataframe):
    """
    Standardize inconsistent HVAC labels.
    """

    dataframe["HVAC Operation Mode"] = (
        dataframe["HVAC Operation Mode"]
        .str.lower()
        .str.strip()
    )

    return dataframe


def fill_missing_values(dataframe):
    """
    Fill missing values using median and mode imputation.
    """

    numerical_columns = [
        "Humidity",
        "MetalOxideSensor_Unit2",
        "CO_GasSensor"
    ]

    for column in numerical_columns:

        median_value = dataframe[column].median()

        dataframe[column] = dataframe[column].fillna(median_value)

    mode_value = dataframe["Ambient Light Level"].mode()[0]

    dataframe["Ambient Light Level"] = (
        dataframe["Ambient Light Level"]
        .fillna(mode_value)
    )

    return dataframe


def prepare_features(dataframe):
    """
    Prepare encoded feature matrix and target variable.
    """

    X = dataframe.drop(columns=["Activity Level"])

    y = dataframe["Activity Level"]

    # Remove Session ID as it is an identifier, not a predictive feature
    if "Session ID" in X.columns:
        X = X.drop(columns=["Session ID"])

    X = pd.get_dummies(X, drop_first=True)

    return X, y


def split_and_scale(X, y):
    """
    Split dataset and apply feature scaling.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled
    )