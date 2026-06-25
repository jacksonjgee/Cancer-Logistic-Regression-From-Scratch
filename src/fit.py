import numpy as np

def sigmoid(z):
	return 1 / (1 + np.exp(-z)) # np.exp(x) calculates e raised to the power of x

def calculate_gradient(theta, X, y):
	m = y.size # m is the number of instances
	return (X.T @ (sigmoid(X @ theta) - y)) / m
# @ is used for matrix multiplication

def gradient_descent(X, y, alpha = 0.01, epochs = 1000, tol = 1e-7):
	X_b = np.c_[np.ones((X.shape[0], 1)), X] # Adds bias/intercept
	theta = np.zeros(X_b.shape[1]) # Parameters initialised to 0's
	
	for i in range(epochs):
		grad = calculate_gradient(theta, X_b, y) # Calculate the gradient
		theta -= alpha * grad # Adjust parameters
		
		if np.linalg.norm(grad) < tol: # tolerance break
			break
		
	return theta # Return the trained parameters

def prediction_probability(X, theta):
	X_b = np.c_[np.ones((X.shape[0], 1)), X]
	return sigmoid(X_b @ theta)
	
def predict_class(X, theta, threshold = 0.5):
	class_pred = (prediction_probability(X, theta) >= threshold)
	return class_pred.astype(int)