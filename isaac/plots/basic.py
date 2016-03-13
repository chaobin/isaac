from contextlib import contextmanager

import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D


__all__ = [
    "zoom_plot",
    "plot",
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

@contextmanager
def d3():
    import mpld3
    mpld3.enable_notebook()
    yield
    mpld3.disable_notebook()

def plot(X, Y, label=None, style='r-', grid=True, title=None, loc=None,
         label_xy=('x', 'y'), show=True):
    if label:
        plt.plot(X, Y, style, label=label)
    else:
        plt.plot(X, Y, style)
    plt.xlabel(label_xy[0])
    plt.ylabel(label_xy[1])
    if title: plt.title(title)
    if loc: plt.legend(loc=loc)
    plt.grid(grid)
    if show: plt.show()

def subplots(h, v=1, order='v', sharex=True, sharey=True,
             plots=()):
    assert (order in ('v', 'vertical', 'h', 'horizontal')), (
        'order must be either vertical or horizontal')
    f, axes = plt.subplots(h, v, sharex=sharex, sharey=sharey)
    def _axes():
        I, J = (h, v) if order == 'v' else (v, h)
        for i in range(I):
            for j in range(J):
                axs = axes[i][j] if order == 'v' else axes[j][i]
                yield axs
    for (axs, (plotter, args, kwargs)) in zip(_axes(), plots):
        plt.axes(axs) # set axs as current active axes
        kwargs['show'] = False
        plotter(*args, **kwargs)
    f.tight_layout(pad=1.3)
    plt.show()

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
