import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + 1e-9
            self.priors[c] = X_c.shape[0]/X.shape[0]

    def _pdf(self, class_idx, X):
        X = np.array(X, dtype=float)
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (X - mean)**2/(2*var))
        denominator = np.sqrt(2*np.pi*var)
        return numerator/denominator
    
    def _predict_sample(self, X):
        posteriors = []

        for c in self.classes:
            prior = np.log(self.priors[c])
            conditional = np.sum(np.log(self._pdf(c, X)))
            posterior = prior + conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        return np.array([self._predict_sample(x) for x in X])