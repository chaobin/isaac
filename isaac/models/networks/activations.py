import numpy as np

__all__ = [
    'Sigmoidal',
    'Softmax'
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
        raise NotImplementedError('')

    @classmethod
    def derivative(cls, z):
        raise NotImplementedError('')
