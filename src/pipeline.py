import os
import joblib
import config

from data_ingestion import load_data

from preprocessing import (
    clean_activity_labels,
    clean_hvac_labels,
    fill_missing_values,
    prepare_features,
    split_and_scale
)

from train_model import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost
)

from evaluate_model import (
    evaluate_model,
    evaluate_xgboost
)


def save_model(model, filename):
    """
    Save a trained model to the saved_model directory.
    """

    os.makedirs(config.SAVED_MODEL_DIR, exist_ok=True)

    filepath = os.path.join(config.SAVED_MODEL_DIR, filename)

    joblib.dump(model, filepath)

    print(f"Model saved to {filepath}")


def main():

    # load dataset
    dataframe = load_data(config.DATABASE_PATH)

    # preprocessing
    dataframe = clean_activity_labels(dataframe)

    dataframe = clean_hvac_labels(dataframe)

    dataframe = fill_missing_values(dataframe)

    # features and target
    X, y = prepare_features(dataframe)

    # split and scale
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled
    ) = split_and_scale(X, y)

    # Logistic Regression
    logistic_model = train_logistic_regression(
        X_train_scaled,
        y_train
    )

    logistic_report, _ = evaluate_model(
        logistic_model,
        X_test_scaled,
        y_test
    )

    print("\nLogistic Regression Results")
    print(logistic_report)

    save_model(logistic_model, "logistic_regression.pkl")

    # Random Forest
    rf_model = train_random_forest(
        X_train,
        y_train
    )

    rf_report, _ = evaluate_model(
        rf_model,
        X_test,
        y_test
    )

    print("\nRandom Forest Results")
    print(rf_report)

    save_model(rf_model, "random_forest.pkl")

    # XGBoost
    xgb_model, label_encoder = train_xgboost(
        X_train,
        y_train
    )

    xgb_report, _ = evaluate_xgboost(
        xgb_model,
        label_encoder,
        X_test,
        y_test
    )

    print("\nXGBoost Results")
    print(xgb_report)

    save_model(xgb_model, "xgboost.pkl")
    save_model(label_encoder, "label_encoder.pkl")


if __name__ == "__main__":
    main()
