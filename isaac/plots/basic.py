import random
from contextlib import contextmanager

import six
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import (
    rcParams,
    colors
)
from mpl_toolkits.mplot3d import Axes3D


__all__ = [
    "zoom_plot",
    "plot",
    "plot_predictions_3d",
    'Palette',
    'plot_clusters',
]


def sorted_color_maps():
    '''List of color name and their hex values sorted by HSV.

    This code is taken from:
        http://matplotlib.org/examples/color/named_colors.html
    '''
    colors_ = list(six.iteritems(colors.cnames))

    # Add the single letter colors.
    for name, rgb in six.iteritems(colors.ColorConverter.colors):
        hex_ = colors.rgb2hex(rgb)
        colors_.append((name, hex_))

    # Transform to hex color values.
    hex_ = [color[1] for color in colors_]
    # Get the rgb equivalent.
    rgb = [colors.hex2color(color) for color in hex_]
    # Get the hsv equivalent.
    hsv = [colors.rgb_to_hsv(color) for color in rgb]

    # Split the hsv values to sort.
    hue = [color[0] for color in hsv]
    sat = [color[1] for color in hsv]
    val = [color[2] for color in hsv]

    # Sort by hue, saturation and value.
    ind = np.lexsort((val, sat, hue))
    sorted_colors = [colors_[i] for i in ind]
    sorted_colors = [
        c_1
        for (c_1, c_2) in zip(sorted_colors[:-1], sorted_colors[1:])
        if c_1[1] != c_2[1]]
    return sorted_colors

class Palette(object):

    SORTED_COLORS = sorted_color_maps()
    GROUPS = (
        #(color_name in SORTED_COLORS, group_name)
        ('k', 'GRAY'),
        ('whitesmoke', 'WHITE'),
        ('rosybrown', 'BROWN'),
        ('firebrick', 'RED'),
        ('sienna', 'SIENNA'),
        ('antiquewhite', 'WHITE'),
        ('orange', 'ORANGE'),
        ('y', 'GREEN'),
        ('mediumaquamarine', 'BLUE'),
        ('mediumpurple', 'PURPLE')
    )

    def __init__(self):
        self.make_palette()

    def make_palette(self):
        group_names = dict(self.GROUPS)
        [setattr(self, grp, []) for (cname, grp) in self.GROUPS]
        current_group = None
        for (cname, ccode) in self.SORTED_COLORS:
            group_name = group_names.get(cname)
            if not (group_name is None):
                current_group = getattr(self, group_name, current_group)
                if current_group is None:
                    continue
            current_group.append(cname)

Palette = Palette()


@contextmanager
def zoom_plot(w, h):
    '''Temprarily change the plot size.

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

def plot_clusters(x, y, k, palette=Palette.GREEN):
    colors = random.sample(palette, k)
    for i in range(k):
        x_i = x[np.nonzero(y==i)]
        plt.scatter(
            x_i[:, 0], x_i[:, 1],
            marker='o', facecolors='none', edgecolors=colors[i])
    plt.grid(True)
    plt.show()
