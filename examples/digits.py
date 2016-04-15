from mnist import MNIST

from isaac.models.networks import forward


PATH_MNIST_DATASET = os.environ['PATH_MNIST_DATA']


def get_training_data():
    mnist = MNIST(PATH_MNIST_DATASET)
    images, labels = mnist.load_training()
    return (images, labels)

def get_testing_data():
    mnist = MNIST(PATH_MNIST_DATASET)
    images, labels = mnist.load_testing()
    return (images, labels)

def train(network):
    pass

def validate(network):
    pass

def main():
    pass
