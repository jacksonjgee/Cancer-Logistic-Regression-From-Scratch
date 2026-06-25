import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from src.fit import sigmoid

def ensure_assets_folder_exists():
    os.makedirs("assets", exist_ok=True)

def plot_cross_entropy_vs_epochs(loss):
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss)

    plt.title("Cross-Entropy Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")

    plt.grid(True)
    plt.savefig("assets/cross-entropy-loss-vs-epochs.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    true_positive = np.sum((y_true == 1) & (y_pred == 1))

    confusion_matrix = np.array([
        [true_negative, false_positive],
        [false_negative, true_positive]
    ])

    plt.figure(figsize=(8, 5))
    plt.imshow(confusion_matrix)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks([0, 1], ["Class 0", "Class 1"])
    plt.yticks([0, 1], ["Class 0", "Class 1"])

    for row in range(2):
        for col in range(2):
            plt.text(
                col,
                row,
                confusion_matrix[row, col],
                ha="center",
                va="center"
            )

    plt.colorbar(label="Count")
    plt.savefig("assets/confusion-matrix.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_predicted_probability_distribution(y_pred_prob):
    plt.figure(figsize=(8, 5))

    plt.hist(y_pred_prob, bins=20, edgecolor="black")

    plt.title("Predicted Probability Distribution")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Number of Examples")

    plt.xlim(0, 1)
    plt.grid(True)
    plt.savefig("assets/predicted-probability-distribution.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def animate_sigmoid_fit(X_feature, y, theta_history, feature_name="Feature"):
    x_min = X_feature.min()
    x_max = X_feature.max()

    x_curve = np.linspace(x_min, x_max, 300)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(X_feature, y, alpha=0.6, label="Training Data")

    line, = ax.plot([], [], label="Sigmoid Curve")
    text = ax.text(0.05, 0.95, "", transform=ax.transAxes)

    ax.set_title(f"Logistic Regression Fit Using {feature_name}")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Predicted Probability")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.1, 1.1)

    ax.grid(True)
    ax.legend()

    def update(frame):
        theta = theta_history[frame]

        bias = theta[0]
        weight = theta[1]

        z = bias + weight * x_curve
        y_curve = sigmoid(z)

        line.set_data(x_curve, y_curve)
        text.set_text(f"Epoch: {frame + 1}")

        return line, text

    frame_step = 5
    frame_indices = range(0, len(theta_history), frame_step)

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=30,
        blit=True
    )

    writer = FFMpegWriter(fps=30)
    animation.save("assets/sigmoid_training_fit.mp4", writer=writer)
    plt.show()
    plt.close()
    