import numpy as np


__all__ = [
    "Network"
]


class Network(object):

    def __init__(self, layering, activation='sigmoid'):
        self.layering = layering
        self.setup(self.layering)

    def setup(self, layering):
        '''
        layering
            list, number of features in each dimension
            e.g., (64 * 64, 64 * 64, 10)
        '''
        self.num_layer = len(layering)
        self.weights = []
        self.biases = []
        for (i, j) in zip(layering[:-1], layering[1:]):
            weight = self.init_weight(rows=j, columns=i)
            self.weights.append(weight)
            bias = self.init_bias(j)
            self.biases.append(bias)

    def init_weight(self, rows, columns):
        '''
        return
            np.ndarray, a [rows] * [columns] matrice
            representing the mapping from one layer
            to the next
        '''
        return np.random.randn(rows, columns)

    def init_bias(self, n):
        '''
        return
            np.ndarray, a vector of biases complementing
            the mapping from one layer to the next.
        '''
        return np.random.randn(n)

    def activation_sigmoid(self, product):
        return 1 / (1 + np.exp(-(product)))

    def derivative_sigmoid(self, product):
        z = self.activation_sigmoid(product)
        return z * (1 - z)

    def activate(self, product, method='sigmoid'):
        activation = 'activation_' + method
        activation = getattr(self, activation)(product)
        return activation

    def activation_derivative(self, product, method='sigmoid'):
        derivative = 'derivative_' + method
        derivative = getattr(self, derivative)(product)
        return derivative

    def SGD(self, Xs, Ys, learning_rate, batch_size, epoch):
        for e in range(epoch):
            batches = [
                (Xs[n:n+batch_size], Ys[n:n+batch_size])
                for n in range(0, len(Ys), batch_size)]
            for (X, Y) in batches:
                self.train_with_n_examples(X, Y, learning_rate)
            print("Epoch {0} completed.".format(e))

    # TODO full vectorization
    def train_with_n_examples(self, Xs, Ys, learning_rate):
        for (x, y) in zip(Xs, Ys):
            g_weights, g_biases = self.backward(x, y)
            # g_weights *= learning_rate
            # g_biases *= learning_rate
            self.update_weights(g_weights, g_biases)

    def forward(self, X):
        '''
        X
            np.array, the preprocessed input. Preprocess may include
            scaling so that the calculations later on it won't overflow.

        return
            list, activations of each neuron
        '''
        products = []
        activations = [X]
        for i in range(self.num_layer-1):
            current_activation = activations[i]
            output = np.dot(
                self.weights[i], current_activation) + self.biases[i]
            # activation with activation function, e.g., sigmoid
            products.append(output)
            next_activation = self.activate(output)
            activations.append(next_activation)
        return (products, activations)

    # TODO full vectorization
    def backward(self, X, Y):
        gradient_weights = [np.zeros_like(w) for w in self.weights]
        gradient_biases = [np.zeros_like(b) for b in self.biases]

        Z, A = self.forward(X)

        # 1. Compute the error
        # (output - y) * activation_derivative(output)
        z = Z[-1] # output of the final layer
        cost_derivative = self.cost_derivative(A[-1], Y)
        delta = cost_derivative * self.activation_derivative(z)
        # compute gradient in curent layer using delta
        gradient_weights[-1] = np.outer(A[-1-1], delta).transpose()
        gradient_biases[-1] = delta
        # 2. Back-propagate the error
        for l in range(2, self.num_layer): # reversed
            delta = np.dot(
                self.weights[-l+1].T, delta) * self.activation_derivative(Z[-l])
            # compute the gradient in current layer using delta
            gradient_weights[-l] = np.outer(A[-l-1], delta).transpose()
            gradient_biases[-l] = delta

        return (gradient_weights, gradient_biases)

    def update_weights(self, weights, biases):
        for i in range(self.num_layer-1):
            self.weights[i] = self.weights[i] - weights[i]
            self.biases[i] = self.biases[i] - biases[i]

    def cost(self, X, Y):
        costs = 0
        for (x, y) in zip(X, Y):
            products, activations = self.forward(x)
            h = activations[-1]
            cost = np.sum(y * np.log(h) + (1 - y) * (np.log(1 - h)))
            costs += cost
        return - (costs / len(X))

    def cost_derivative(self, output, Y):
        '''
        The cost derivative with respect to the output
        in the final layer.
        '''
        return (output - Y)

    def predict(self, x):
        _, activations = self.forward(x)
        return np.argmax(activations[-1])

    def accuracy(self, Xs, Ys):
        score = 0
        for (x, y) in zip(Xs, Ys):
            answer = self.predict(x)
            if answer == y:
                score += 1
        return score / len(Xs)




