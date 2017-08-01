import numpy as np

# generate sample data
np.random.seed(1)
data_points = 1000

pos_mean = [0, 1]
neg_mean = [1, 4]
cov = [[1, .7], [.7, 1]]

pos_data = np.random.multivariate_normal(pos_mean, cov, data_points)

neg_data = np.random.multivariate_normal(neg_mean, cov, data_points)

data_features = np.vstack((pos_data, neg_data)).astype(np.float32)

data_labels = np.hstack((np.zeros(data_points), np.ones(data_points)))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Sum of the log of the Probability that data_point(x) produces target(y) given weight(w)
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


def logistic_regression(features, target, iterations, learning_rate):
    # Initial array for weights
    weights = np.zeros(features.shape[1])

    for i in range(iterations):
        scores = np.dot(features, weights)
        prediction = sigmoid(scores)

        g = gradient(features, target, prediction)
        weights += learning_rate * g

        # Print log-likelihood every so often
        if i % 1000 == 0:
            print("LL", log_likelihood(features, target, weights))

    return weights

weights = logistic_regression(data_features, data_labels, iterations=10000, learning_rate=0.01)