from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def evaluate_model(model, X_test, y_test):
    """
    Generate classification report for model evaluation.
    """

    predictions = model.predict(X_test)

    report = classification_report(y_test, predictions)

    matrix = confusion_matrix(y_test, predictions)

    return report, matrix


def evaluate_xgboost(
    model,
    label_encoder,
    X_test,
    y_test
):
    """
    Evaluate XGBoost model with decoded labels.
    """

    predictions = model.predict(X_test)

    decoded_predictions = (
        label_encoder.inverse_transform(predictions)
    )

    report = classification_report(
        y_test,
        decoded_predictions
    )

    matrix = confusion_matrix(
        y_test,
        decoded_predictions
    )

    return report, matrix