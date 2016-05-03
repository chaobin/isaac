__all__ = [
    "Descent"
]


class Descent(object):

    def __init__(self, model, X, Y, rate=0.1,
        stochastic=False, size_batch=None):
        self.model = model
        self.X = X
        self.Y = Y
        self.rate = 0.1
        self.stochastic = stochastic
        self.size_batch = size_batch

    def run(self, distance=100, rate=None):
        converged = False
        rate = rate or self.rate
        while not converged:
            x, y = self.update_batch() if self.stochastic else (self.X, self.Y)
            changes = self.model.cost_derivative(x, y)
            weights_updated = (self.model.weights - rate * changes) # resize the step
            converged = (weights_updated == self.model.weights).all()
            self.model.update_weights(weights_updated)
            distance -= 1
            if distance == 0: # reached checkpoint
                break

    def update_batch(self, size=None):
        raise NotImplementedError()

