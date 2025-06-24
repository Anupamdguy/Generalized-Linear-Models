import numpy as np
from sklearn.preprocessing import StandardScaler

class LinearRegressionScratch:

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dW = -(2/n_samples) * np.dot(X.T, (y - y_pred))
            db = -(2/n_samples) * np.sum(y - y_pred)

            self.weights -= self.lr * dW
            self.bias -= self.lr * db

    def predict(self, X):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return np.dot(X, self.weights) + self.bias