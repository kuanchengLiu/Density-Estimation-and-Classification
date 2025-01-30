import numpy as np
import scipy.io
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

# Load the MNIST dataset
data = scipy.io.loadmat('mnist_data.mat')

# Check the keys in the loaded .mat file
print("Keys in dataset:", data.keys())

# Extract training and testing data
trX = data['trX']
trY = data['trY']
tsX = data['tsX']
tsY = data['tsY']

# Ensure trY is a 1D array (flatten it if necessary)
trY = trY.flatten()
tsY = tsY.flatten()

# Re-filter the training and testing data for digits 7 (0) and 8 (1)
trY_filtered = trY[(trY == 0) | (trY == 1)]
tsY_filtered = tsY[(tsY == 0) | (tsY == 1)]

# Re-filter the features for digits 7 (0) and 8 (1)
trX_filtered = trX[(trY == 0) | (trY == 1)]
tsX_filtered = tsX[(tsY == 0) | (tsY == 1)]

# Scale the data using StandardScaler
scaler = StandardScaler()
trX_normalized = scaler.fit_transform(trX_filtered)
tsX_normalized = scaler.transform(tsX_filtered)

# Compute the mean and covariance for each class (7 and 8)
X_7 = trX_normalized[trY_filtered == 0]
X_8 = trX_normalized[trY_filtered == 1]

mean_7 = np.mean(X_7, axis=0)
cov_7 = np.cov(X_7, rowvar=False)
mean_8 = np.mean(X_8, axis=0)
cov_8 = np.cov(X_8, rowvar=False)

# Add a small regularization term to the covariance matrices to make them positive definite
epsilon = 1e-3  # Increased regularization term
cov_7_reg = cov_7 + np.eye(cov_7.shape[0]) * epsilon
cov_8_reg = cov_8 + np.eye(cov_8.shape[0]) * epsilon

# Logistic Regression using gradient ascent
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = X.shape[0]
    predictions = sigmoid(np.dot(X, theta))
    return -(1/m) * (np.dot(y, np.log(predictions)) + np.dot(1 - y, np.log(1 - predictions)))

def gradient(X, y, theta):
    m = X.shape[0]
    predictions = sigmoid(np.dot(X, theta))
    return (1/m) * np.dot(X.T, predictions - y)

def gradient_ascent(X, y, theta, learning_rate=0.1, iterations=1000):
    cost_history = []
    for _ in range(iterations):
        grad = gradient(X, y, theta)
        theta = theta - learning_rate * grad
        cost_history.append(cost_function(X, y, theta))
    return theta, cost_history

# Add intercept (bias) term to training data
X_train_with_intercept = np.hstack([np.ones((trX_normalized.shape[0], 1)), trX_normalized])
X_test_with_intercept = np.hstack([np.ones((tsX_normalized.shape[0], 1)), tsX_normalized])

# Initialize theta to zeros
theta_initial = np.zeros(X_train_with_intercept.shape[1])

# Train the Logistic Regression model using gradient ascent
theta_optimal, cost_history = gradient_ascent(X_train_with_intercept, trY_filtered, theta_initial, learning_rate=0.01, iterations=500)

# Predict on the test set using Logistic Regression
predictions_logistic_regression = sigmoid(np.dot(X_test_with_intercept, theta_optimal)) >= 0.5

# Calculate accuracy for digits 7 and 8
accuracy_logistic_regression = np.mean(predictions_logistic_regression == tsY_filtered)

# Naïve Bayes Classification using the regularized covariance matrices
def naive_bayes_predict(X, mean_7, cov_7, mean_8, cov_8):
    log_prob_7 = multivariate_normal.logpdf(X, mean=mean_7, cov=cov_7)
    log_prob_8 = multivariate_normal.logpdf(X, mean=mean_8, cov=cov_8)
    return (log_prob_8 > log_prob_7).astype(int)

# Predict on the test set using Naïve Bayes
predictions_naive_bayes_reg = naive_bayes_predict(tsX_normalized, mean_7, cov_7_reg, mean_8, cov_8_reg)

# Calculate accuracy for Naïve Bayes
accuracy_naive_bayes_reg = np.mean(predictions_naive_bayes_reg == tsY_filtered)

# Output results
print(f"Logistic Regression Accuracy: {accuracy_logistic_regression * 100:.2f}%")
print(f"Naïve Bayes Accuracy: {accuracy_naive_bayes_reg * 100:.2f}%")