__all__ = [
    "batch",
    "stochastic"
]


class Descent(object):

    def __init__(self, model, X, Y, rate=0.1):
        self.model = model
        self.inputs = X
        self.outputs = Y
        self.rate = 0.1

    def run(self, distance=100, rate=None):
        converged = False
        rate = rate or self.rate
        while not converged:
            X, Y = self.update_batch()
            changes = self.model.cost_derivative(X, Y)
            weights_updated = (self.model.weights - rate * changes) # resize the step
            converged = (weights_updated == self.model.weights).all()
            self.model.update_weights(weights_updated)
            distance -= 1
            if distance == 0: # reached checkpoint
                break

    def update_batch(self, size=None):
        return (self.inputs, self.outputs)

