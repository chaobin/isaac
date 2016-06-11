from functools import partial

import graphviz as gv


__all__ = [
    'graph',
    'digraph'
]


graph = partial(gv.Graph, format='png')
digraph = partial(gv.Digraph, format='png')

def add_nodes(graph, nodes):
    for n in nodes:
        if isinstance(n, tuple):
            graph.node(n[0], **n[1])
        else:
            graph.node(n)
    return graph

def add_edges(graph, edges):
    for e in edges:
        if isinstance(e[0], tuple):
            graph.edge(*e[0], **e[1])
        else:
            graph.edge(*e)
    return graph

def apply_styles(graph, styles):
    graph.graph_attr.update(
        styles.get('graph', {})
    )
    graph.node_attr.update(
        styles.get('nodes', {})
    )
    graph.edge_attr.update(
        styles.get('edges', {})
    )
    return graph
