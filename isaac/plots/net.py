import os
from itertools import product
from contextlib import contextmanager

import graphviz as gv
from isaac.plots import dot


__all__ = [
    'folder',
    'forward_net'
]

# used to render the svg
__NAME = None

# http://www.graphviz.org/doc/info/attrs.html
nn_styling = {
    'graph': {
        'color': 'white', # Hide the cluster border
        'fontsize': '9',
        # 'bgcolor': '#333333',
        'rankdir': 'LR',
        # 'rank': 'same',
        'ranksep': 'equally',
        'splines': 'line',
        'fixedsize': 'true',
        'ratio': '1',
        'center': 'true',
        'start': 'random',
        'nodesep': '.05'
    },
    'nodes': {
        'shape': 'circle',
        'width': '0.25',
        'height': '0.25',
        'penwidth': '0.5',
        # 'fontsize': '8',
        # 'style': 'dashed',
    },
    'edges': {
        'width': '0.01',
        # 'arrowhead': 'open',
        'arrowtail': 'open',
        'arrowsize': '0.2',
        'weight': '0.5',
        'penwidth': '0.5',
        'weight': '100', # !IMPORTANT centering the node in cluster
    }
}


@contextmanager
def folder(root, ext='png'):
    global __NAME
    if not os.path.exists(root): os.mkdir(root)
    cursor = os.path.join(root, 'next')
    name = None
    if not os.path.exists(cursor):
        with open(cursor, 'w') as f:
            name = str(0)
            f.write(name)
    else:
        with open(cursor) as f:
            name = f.read()
    __NAME = os.path.join(root, os.path.extsep.join((name, ext)))
    yield
    name = int(name)
    name += 1
    with open(cursor, 'w') as f:
        f.write(str(name))
    __NAME = None

def forward_net(layering=(),
    labels=('input', 'hidden', 'output'), reversed=False):
    label_i, label_h, label_o = labels
    root = gv.Digraph(engine='dot')
    last_layer = None

    def connect(l1, l2):
        edges = product(l1, l2)
        return edges

    for (index, l) in enumerate(layering):
        first = (index == 0)
        last = (index == len(layering) - 1)
        labelled_nodes = [
                ('%s%s' % (index, i), {'label': ''})
                # ('%s%s' % (index, i), {'label': '%s%s' % (index, i)}) # show label
                for i in range(l)]
        nodes = ['%s%s' % (index, i) for i in range(l)]
        g = gv.Digraph('cluster_%s' % index)
        # add node
        dot.add_nodes(g, labelled_nodes)
        if not (last_layer is None):
            edges = connect(last_layer, nodes)
            dot.add_edges(root, edges)
        if first: g.body.append('label=%s' % label_i)
        elif last: g.body.append('label=%s' % label_o)
        else: g.body.append('label=%s' % label_h)
        last_layer = nodes
        root.subgraph(g)

    dot.apply_styles(root, nn_styling)
    if reversed:
        root.edge_attr.update({'dir': 'both', 'arrowhead': 'none', 'arrowtail': 'open'})
    # root.body.extend(["%s=%s" % (k, v) for (k, v) in nn_styling['graph'].items()])
    # Stores this so that the nvconvert will have it
    root.render(__NAME, view=True)
    return root
