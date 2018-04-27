#!/usr/bin/env python

import networkx as nx

def partly_relabel_by_sorted_attr(g_old, select_attr, select_attr_vals, sort_attr):
    """
    Relabel nodes of NetworkX graph such that the new IDs are
    sorted according to the order of some attribute such that 
    only the nodes that have that attribute are relabeled.
                           
    Parameters
    ----------
    g_old : networkx.MultiDiGraph
        NetworkX graph.
    select_attr : object 
        Attribute to examine to determine which nodes 
        to relabel. If a node does not have this attribute, it is
        not relabeled.
    select_attr_vals : list
        If `select_attr` is set to any of these values, it is relabeled. If it
        is None, any value is accepted.
    sort_attr : object
        Attribute on which to sort nodes that are to be relabeled.

    Returns
    -------
    g_new : networkx.MultiDiGraph
        Graph containing partly relabeled nodes.
    """

    assert isinstance(g_old, nx.MultiDiGraph)
    assert isinstance(select_attr_vals, list) or select_attr_vals is None

    # Only sort nodes with `select_attr` attribute:
    if select_attr_vals is None:
        nodes_to_sort = [n for n in g_old.nodes(True) \
                         if select_attr in n[1]]
    else:
        nodes_to_sort = [n for n in g_old.nodes(True) \
                         if select_attr in n[1] and n[1][select_attr] in select_attr_vals]
    nodes_to_ignore = [n for n in g_old.nodes(True) if select_attr not in n[1]]

    # Sort nodes by value of `sort_attr` attribute:
    nodes_sorted = sorted(nodes_to_sort, key=lambda n: n[1][sort_attr])

    # First, add the unsorted nodes whose IDs should remain the same:
    mapping = {}
    for n in nodes_to_ignore:
        mapping[n[0]] = n[0]

    # Next, create mapping between old IDs and sorted IDs that makes the order
    # of the node IDs correspond to that of the `sort_attr` values:
    for n, m in zip(nodes_to_sort, nodes_sorted): 
        mapping[m[0]] = n[0]

    return nx.relabel_nodes(g_old, mapping)
