from contextlib import contextmanager

import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D


__all__ = [
    "zoom_plot",
    "plot_predictions_3d",
]


@contextmanager
def zoom_plot(w, h):
    '''
    Temprarily change the plot size.
    '''
    shape = rcParams['figure.figsize']
    rcParams['figure.figsize'] = w, h
    yield
    rcParams['figure.figsize'] = shape

def plot_predictions_3d(X, Y, predictions, labels,
                  mirror=False,
                  title=""):
    '''
    Plot the [predictions] against the output [Y] projected by
    two X.
    '''
    assert len(labels) == 2, "we are only plotting a 3D projection with 2 features"
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # plot the reality
    f1, f2 = labels
    x1, x2 = X[:, 0], X[:, 1]
    if mirror:
        f1, f2 = f2, f1
        x1, x2 = x2, x1
    ax.scatter(x1, x2, Y, c='r', marker='o', label='actual univ GPA')

    # plot the predition
    ax.scatter(x1, x2, predictions, c='g', label='predicted univ GPA')

    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_zlabel('prediction VS. example')
    
    plt.title(title)
    plt.legend()
    plt.show()
