from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.fit import (
    gradient_descent,
    predict_class,
    prediction_probability
)

from src.plots import (
    plot_cross_entropy_vs_epochs,
    plot_confusion_matrix,
    plot_predicted_probability_distribution,
    animate_sigmoid_fit,
    ensure_assets_folder_exists
)


def run():
    data = load_breast_cancer()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=67
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    theta, loss_history, theta_history = gradient_descent(
        X_train_scaled,
        y_train,
        alpha=0.01,
        epochs=1000
    )

    y_pred_train = predict_class(X_train_scaled, theta)
    y_pred_test = predict_class(X_test_scaled, theta)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    ensure_assets_folder_exists()

    plot_cross_entropy_vs_epochs(loss_history)
    plot_confusion_matrix(y_test, y_pred_test)

    y_pred_prob_test = prediction_probability(X_test_scaled, theta)
    plot_predicted_probability_distribution(y_pred_prob_test)

    # Separate 1-feature model for the sigmoid animation
    feature_index = 0
    feature_name = data.feature_names[feature_index]

    X_one_feature = X[:, feature_index].reshape(-1, 1)

    X_train_one, X_test_one, y_train_one, y_test_one = train_test_split(
        X_one_feature, y, test_size=0.2, random_state=67
    )

    one_feature_scaler = StandardScaler()

    X_train_one_scaled = one_feature_scaler.fit_transform(X_train_one)

    theta_one, loss_history_one, theta_history_one = gradient_descent(
        X_train_one_scaled,
        y_train_one,
        alpha=0.01,
        epochs=1000
    )

    animate_sigmoid_fit(
        X_train_one_scaled.flatten(),
        y_train_one,
        theta_history_one,
        feature_name=f"Scaled {feature_name}"
    )