import numpy as np

def sigmoid(z):
	z = np.clip(z, -500, 500)
	return 1.0 / (1.0 + np.exp(-z)) # np.exp(x) calculates e raised to the power of x

def calculate_gradient(theta, X, y):
	m = y.size # m is the number of instances
	return (X.T @ (sigmoid(X @ theta) - y)) / m
# @ is used for matrix multiplication

def cross_entropy_loss(y, y_pred):
	epsilon = 1e-15
	y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

	total_loss = 0
	m = y.size
	
	for i in range(m):
		current_loss = y[i] * np.log(y_pred[i]) + (1.0 - y[i]) * np.log(1.0 - y_pred[i])
		total_loss += current_loss
		
	return -(total_loss / m)

def gradient_descent(X, y, alpha = 0.01, epochs = 2000, tol = 1e-7):
	X_b = np.c_[np.ones((X.shape[0], 1)), X] # Adds bias/intercept
	theta = np.zeros(X_b.shape[1]) # Parameters initialised to 0's

	loss_history = []
	theta_history = []

	for i in range(epochs):
		y_pred = sigmoid(X_b @ theta)

		loss = cross_entropy_loss(y, y_pred)
		loss_history.append(loss)

		grad = calculate_gradient(theta, X_b, y) # Calculate the gradient
		theta -= alpha * grad # Adjust parameters
		
		theta_history.append(theta.copy())

		if np.linalg.norm(grad) < tol: # tolerance break
			break
		
	return theta, loss_history, theta_history # Return the trained parameters, loss history, and theta history

def prediction_probability(X, theta):
	X_b = np.c_[np.ones((X.shape[0], 1)), X]
	return sigmoid(X_b @ theta)
	
def predict_class(X, theta, threshold = 0.5):
	class_pred = (prediction_probability(X, theta) >= threshold)
	return class_pred.astype(int)