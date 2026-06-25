from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.fit import gradient_descent, predict_class
from sklearn.preprocessing import StandardScaler

def run():
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    theta = gradient_descent(X_train_scaled, y_train)
    y_pred = predict_class(X_test_scaled, theta)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)