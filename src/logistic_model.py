import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def fit(self, X, y):
        n_examples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_output = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_output)

            dW = (1/n_examples) * np.dot(X.T, (predictions - y))
            db = (1/n_examples) * np.sum(predictions - y)

            self.weights -= self.lr * dW
            self.bias -= self.lr * db

    def predict_prob(self, X):
        return self._sigmoid(np.dot(X, self.weights) + self.bias)
    
    def predict(self, X):
        return self.predict_prob(X) >= 0.5