import numpy as np

# generate sample data
np.random.seed(1)
data_points = 1000

pos_mean = [0, 1]
neg_mean = [1, 4]
cov = [[1, .5], [.5, 1]]

pos_data = np.random.multivariate_normal(pos_mean, cov, data_points)

neg_data = np.random.multivariate_normal(neg_mean, cov, data_points)

features = np.vstack((pos_data, neg_data)).astype(np.float32)

labels = np.hstack((np.zeros(data_points), np.ones(data_points)))

