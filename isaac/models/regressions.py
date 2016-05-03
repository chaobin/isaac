import numpy as np

from isaac.stats import sigmoid


__all__ = [
    "LinearRegression",
    "LogisticRegression"
]


class LinearRegression(object):
    
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

    def predict(self, x):
        return np.dot(x, self.weights)
    
    def costs(self, x, y):
        '''
        Measuring the Mean Squared Error over the training set.
        '''
        return np.mean(np.power(np.dot(x, self.weights) - y, 2))

    def cost_derivative(self, x, y):
        costs = (np.dot(x, self.weights) - y)
        derivatives = np.mean(x.T * costs, axis=1)
        return derivatives

    def update_weights(self, new_weights):
        self.weights = new_weights


class LogisticRegression(LinearRegression):
    
    def predict(self, x):
        return 1 / (1 + np.exp((- np.dot(x, self.weights))))

    def one_cost(self, x, y):
        prediction = self.predict(Xs)
        cost = (
            (- y) * math.log(prediction) - (1 - y) * (math.log(1 - prediction))
            )
        return cost

    def costs(self, x, y):
        predictions = self.predict(x)
        # -log(self.predict(X) if Y == 1)
        # -log(1 - self.predict(X) if Y == 0)
        return np.mean(
            (- y) * np.log(predictions) - (1 - y) * (np.log(1 - predictions)))

    def cost_derivative(self, x, y):
        costs = (self.predict(x) - y)
        derivatives = np.mean(x.T * costs, axis=1)
        return derivatives

    # TODO I suspect this implementation here below
    # is not a clever one.
    def accuracy(self, x, y):
        '''
        return
            float, percentage of accuracy
        '''
        total = len(y)
        predictions = self.predict(x)
        # step the values
        predictions[predictions > 0.5] = 1
        predictions[predictions < 0.5] = 0
        # stats the accuracy
        validations = (predictions == y)
        return validations.mean()

