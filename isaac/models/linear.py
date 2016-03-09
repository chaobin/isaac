import numpy as np

from isaac.stats import sigmoid


__all__ = [
    "Regression",
    "LogisticRegression"
]


class Regression(object):
    
    def __init__(self, weights):
        self.weights = weights
        self.dimension = len(weights)
    
    @classmethod
    def from_dimension(cls, dimension, value=None, dtype=None):
        dtype = dtype or np.float64
        if value is None:
            weights = np.ones(dimension, dtype)
        else:
            weights = np.zeros(dimension, dtype) + value
        return cls(weights)

    def predict(self, X):
        return np.dot(X, self.weights)
    
    def cost(self, X, Y):
        '''
        Measuring the Mean Squared Error over the training set.
        '''
        return np.mean(np.power(np.dot(X, self.weights) - Y, 2))

    def cost_derivative(self, X, Y):
        costs = (np.dot(X, self.weights) - Y)
        derivatives = np.mean(X.T * costs, axis=1)
        return derivatives

    def update_weights(self, new_weights):
        self.weights = new_weights


class LogisticRegression(Regression):
    
    def predict(self, X):
        return sigmoid(np.dot(X, self.weights))

    def one_cost(self, x, y):
        prediction = self.predict(Xs)
        cost = (
            (- y) * math.log(prediction) - (1 - y) * (math.log(1 - prediction))
            )
        return cost

    def cost(self, X, Y):
        '''
        Measuring the Mean Squared Error over the training set.
        '''
        dot_products = np.dot(X, self.weights)
        # squash using sigmoid
        predictions = 1 / (1 + np.power(np.e, (- dot_products)))
        return np.mean(
            (- Y) * np.log(predictions) - (1 - Y) * (np.log(1 - predictions))
        )

