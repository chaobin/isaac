import numpy as np


__all__ = [
    "Network"
]


class Network(object):

    def __init__(self, layering, activation='sigmoid'):
        self.layering = layering
        self.k = layering[-1] # num of classes
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

    def derivative_activation_sigmoid(self, product):
        z = self.activation_sigmoid(product)
        return z * (1 - z)

    def activate(self, product, method='sigmoid'):
        activation = 'activation_' + method
        activation = getattr(self, activation)(product)
        return activation

    def activation_derivative(self, product, method='sigmoid'):
        derivative = 'derivative_activation_' + method
        derivative = getattr(self, derivative)(product)
        return derivative

    def SGD(self, Xs, Ys, lrate, batch_size, epoch):
        size_training_set = len(Ys)
        for e in range(epoch):
            batches = [
                (Xs[n:n+batch_size], Ys[n:n+batch_size])
                for n in range(0, size_training_set, batch_size)]
            for (X, Y) in batches:
                # start training with the batch
                Z, A = self.forward(X)
                d_W, d_B = self.backward(Z, A, Y)
                # average the accumulated gradient
                for i in range(self.num_layer - 1):
                    d_W[i] = lrate * (d_W[i] / len(Y))
                    d_B[i] = lrate * (d_B[i] / len(Y))
                self.update_weights(d_W, d_B)
            print("Epoch {0} completed.".format(e))

    def forward(self, X):
        '''
        X
            np.array, the preprocessed input. Preprocess may include
            scaling so that the calculations later on it won't overflow.

        return
            list, activations of each neuron
        '''
        X = np.atleast_2d(X)
        products = []
        activations = [X]
        for i in range(self.num_layer-1):
            current_activation = activations[i]
            # HINT (VECTORIZATION) np.dot(A, w.T) == np.dot(w, A.T).T
            output = np.dot(
                current_activation, self.weights[i].T) + self.biases[i]
            # activation with activation function, e.g., sigmoid
            products.append(output)
            next_activation = self.activate(output)
            activations.append(next_activation)
        # HINT (products[-1].shape == activations[-l].shape == (len(X), self.k))
        return (products, activations)

    def backward(self, Z, A, Y):
        gradient_weights = [None] * (len(self.weights))
        gradient_biases = [None] * (len(self.biases))
        # 1. Compute the delta of error with respect to the
        #    the output of the final layer
        cost_derivative = self.cost_derivative(A[-1], Y)
        delta = cost_derivative * self.activation_derivative(Z[-1])
        # HINT delta.shape == (len(X), self.k)
        # compute gradient in curent layer using delta
        gradient_weights[-1] = np.dot(delta.T, A[-1-1])
        gradient_biases[-1] = delta.sum(0)
        # 2. Back-propagate the error
        for l in range(2, self.num_layer): # reversed
            # HINT (VECTORIZATION)
            #   The delta comes as a matrix when BP
            #   is fully vectorized. It's easy to see
            #   the transformation in these two steps:
            #   given:
            #       w = matrix(m, i) 
            #       delta = matrix(j, m)
            #   where:
            #       - m is the number of output neurons
            #       - i is number of activating neurons
            #       - j is batch size
            #   you have:
            #       np.dot(w.T, delta[0]) when taking on one example
            #       np.dot(delta, w) when taking on a set of examples
            delta = np.dot(
                delta, self.weights[-l+1]) * self.activation_derivative(Z[-l])
            # compute the gradient in current layer using delta
            # HINT (VECTORIZATION)
            #
            # for i in range(num_of_batch):
            #   d_weights += np.outer(A[-l-1][i], delta[i])
            #   d_biases += delta[i]
            # =>
            # np.dot(A[-l-1].T, delta).T
            # =>
            # np.dot(delta.T, A[-l-1])
            gradient_weights[-l] = np.dot(delta.T, A[-l-1])
            gradient_biases[-l] = delta.sum(0)

        return (gradient_weights, gradient_biases)

    def update_weights(self, weights, biases):
        for i in range(self.num_layer-1):
            self.weights[i] = self.weights[i] - weights[i]
            self.biases[i] = self.biases[i] - biases[i]

    def cost(self, X, Y):
        _, A = self.forward(X)
        H = A[-1]
        cost = np.mean(
            # HINT mean(sum_example(sum_of_class(example)))
            (Y * np.log(H) + (1 - Y) * (np.log(1 - H))).sum(1)
        )
        return -cost

    def cost_derivative(self, output, Y):
        '''
        The cost derivative with respect to the output
        in the final layer.
        '''
        return (output - Y)

    def predict(self, X):
        _, A = self.forward(np.atleast_2d(X))
        return np.argmax(A[-1], axis=1)

    def accuracy(self, X, Y):
        predictions = self.predict(X)
        return (predictions == np.argmax(Y, axis=1)).mean()


