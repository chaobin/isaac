import math


__all__ = [
    "sigmoid"
]

def sigmoid(n):
    return 1 / (1 + pow(math.e, -n))
