import numpy as np

from isaac.models.networks import activations

__all__ = [
    "Network"
]


class Network(object):

    def __init__(self, layering, activation=activations.Sigmoidal):
        self.layering = layering
        self.k = layering[-1] # num of classes
        self.init_weights(self.layering)
        self.activation = activation

    def reset(self):
        self.init_weights(self.layering)

    def init_weights(self, layering):
        '''Init weight in each layer.
        
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
        '''Initialize the weights and biases for one layer.

        Parameters
        ----------

        rows, columns : int
            rows is number of neuron in the next layer,
            columns is the number of output neuron

        Returns
        -------
        np.ndarray
            A [rows] * [columns] matrice representing
            the mapping from one layer to the next.
        '''
        return np.random.randn(rows, columns)

    def init_bias(self, n):
        '''
        return
            np.ndarray, a vector of biases complementing
            the mapping from one layer to the next.
        '''
        return np.random.randn(n)

    def activate(self, z):
        a = self.activation.activate(z)
        return a

    def activation_derivative(self, z):
        dz = self.activation.derivative(z)
        return dz

    def SGD(self, X, Y, lrate, batch_size, epoch):
        size_training_set = len(Y)
        for e in range(epoch):
            batches = [
                (X[n:n+batch_size], Y[n:n+batch_size])
                for n in range(0, size_training_set, batch_size)]
            for (x, y) in batches:
                # start training with the batch
                z, a = self.forward(x)
                d_w, d_b = self.backward(z, a, y)
                # average the accumulated gradient
                for i in range(self.num_layer - 1):
                    d_w[i] = lrate * (d_w[i] / len(y))
                    d_b[i] = lrate * (d_b[i] / len(y))
                self.update_weights(d_w, d_b)
            print("Epoch {0} completed.".format(e))

    def forward(self, x):
        '''Calculate the linear combinations and apply activation.

        Parameters
        ----------

        x : np.array
            The preprocessed input. Preprocess may include
            scaling so that the calculations later on it won't overflow.

        Returns
        -------
        tuple
            (z, a), where z is the linear combinations in each layer,
            and a is the activation using self.activation.
        '''
        x = np.atleast_2d(x)
        products = []
        activations = [x]
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

    def backward(self, z, a, y):
        gradient_weights = [None] * (len(self.weights))
        gradient_biases = [None] * (len(self.biases))
        # 1. Compute the delta of error with respect to the
        #    the output of the final layer
        delta = self.cost_derivative(a[-1], y)
        # HINT delta.shape == (len(X), self.k)
        # compute gradient in curent layer using delta
        gradient_weights[-1] = np.dot(delta.T, a[-1-1])
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
                delta, self.weights[-l+1]) * self.activation_derivative(z[-l])
            # compute the gradient in current layer using delta
            # HINT (VECTORIZATION)
            #
            # for i in range(num_of_batch):
            #   d_weights += np.outer(a[-l-1][i], delta[i])
            #   d_biases += delta[i]
            # =>
            # np.dot(A[-l-1].T, delta).T
            # =>
            # np.dot(delta.T, a[-l-1])
            gradient_weights[-l] = np.dot(delta.T, a[-l-1])
            gradient_biases[-l] = delta.sum(0)

        return (gradient_weights, gradient_biases)

    def update_weights(self, weights, biases):
        for i in range(self.num_layer-1):
            self.weights[i] = self.weights[i] - weights[i]
            self.biases[i] = self.biases[i] - biases[i]

    def cost(self, x, y):
        _, a = self.forward(x)
        h = a[-1]
        cost = np.mean(
            # HINT mean(sum_example(sum_of_class(example)))
            (y * np.log(h) + (1 - y) * (np.log(1 - h))).sum(1)
        )
        return -cost

    def cost_derivative(self, output, y):
        '''Return the delta of the final layer.

        The cost delta with respect to the output
        of the final layer. Here the cost is defined
        using the negative log likelihood or cross entropy.
        The larger the difference between estimation and
        sample is, the larger the error is fedback to network.
        '''
        return (output - y)

    def predict(self, x):
        _, a = self.forward(np.atleast_2d(x))
        return np.argmax(a[-1], axis=1)

    def accuracy(self, x, y):
        predictions = self.predict(x)
        return (predictions == np.argmax(y, axis=1)).mean()

    def mistakes(self, x, y):
        predictions = self.predict(x)
        mistakes = (predictions == np.argmax(y, axis=1))
        return np.nonzero(mistakes==False)[0]

