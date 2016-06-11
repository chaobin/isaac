import numpy as np

__all__ = [
    'Sigmoidal',
    'Softmax',
    'Tanh',
    'ReLU',
]


class Sigmoidal(object):

    @classmethod
    def activate(cls, z):
        return 1 / (1 + np.exp(-(z)))

    @classmethod
    def derivative(cls, z):
        a = cls.activate(z)
        return a * (1 - a)

class Softmax(object):

    @classmethod
    def activate(cls, z):
        return np.exp(z) / np.sum(np.exp(z))

class Tanh(object):

    @classmethod
    def activate(cls, z):
        return np.tanh(z)

    @classmethod
    def derivative(cls, z):
        return 1 - (np.tanh(z))**2

class ReLU(object):

    @classmethod
    def activate(cls, z):
        '''
        g(z) = max(0, z)
        '''
        zeros = np.zeros((len(z)))
        _z = np.column_stack((z, zeros))
        return np.max(_z, axis=1)

    @classmethod
    def derivative(cls, z):
        pass
