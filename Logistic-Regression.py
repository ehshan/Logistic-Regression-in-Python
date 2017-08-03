import numpy as np
import matplotlib.pyplot as plt

# generate sample data
np.random.seed(1)
data_points = 1000

pos_mean = [0, 1]
neg_mean = [1, 4]
cov = [[1, .7], [.7, 1]]

# Mock posiive class
pos_data = np.random.multivariate_normal(pos_mean, cov, data_points)
# Mock negative class
neg_data = np.random.multivariate_normal(neg_mean, cov, data_points)
# All data points
data_features = np.vstack((pos_data, neg_data)).astype(np.float32)
# Split labels between classes
data_labels = np.hstack((np.zeros(data_points), np.ones(data_points)))

# Visualise generated data
plt.figure(figsize=(12, 8))
plt.scatter(data_features[:, 0], data_features[:, 1], c=data_labels, alpha=.5)
plt.show()


# Link function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Sum of the log of the Probability that data_point(x) produces target(y) given weight(w)
# Maximisation function
def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    # Log of the product = the sum of the logs
    l_l = np.sum(target * scores - np.log(1 + np.exp(scores)))
    return l_l


# Gradient of LogLikelihood = features(X) * (target(Y) - Predictions)
def gradient(features, target, predictions):
    error = target - predictions
    g = np.dot(features.T, error)
    return g


# Main function
def logistic_regression(features, target, iterations, learning_rate, intercept=False):
    # For when predictor is 0
    if intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    # Initial array for weights
    weights = np.zeros(features.shape[1])

    for i in range(iterations):
        scores = np.dot(features, weights)
        prediction = sigmoid(scores)

        g = gradient(features, target, prediction)
        weights += learning_rate * g

        if i % 1000 == 0:
            print("Iteration:", i, "Log Likelihood:", log_likelihood(features, target, weights))

    return weights

# Weights that maximise correct classification
weights = logistic_regression(data_features, data_labels, iterations=10000, learning_rate=0.01, intercept=True)

# Calculate the accuracy of model
# All data
data_with_intercept = np.hstack((np.ones((data_features.shape[0], 1)), data_features))
# Compute the outputs
model_scores = np.dot(data_with_intercept, weights)
# Assign output a class
predictions = np.round(sigmoid(model_scores))

# Print results
print('Model Accuracy: {0}'.format((predictions == data_labels).sum().astype(float) / len(predictions)))

# Visualise the model results
plt.figure(figsize=(12, 8))
plt.scatter(data_features[:, 0], data_features[:, 1], c=predictions == data_labels - 1, alpha=.5)
plt.show()
