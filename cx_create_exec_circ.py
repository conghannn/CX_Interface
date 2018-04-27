#!/usr/bin/env python

"""
Create executable circuit nodes/edges.
"""

import itertools
import logging
import sys

import numpy as np
from pyorient.ogm import Graph, Config

import neuroarch.models as models
import neuroarch.query as query
import neuroarch.nxtools as nxtools

from cx_config import cx_db

graph = Graph(Config.from_url(cx_db, 'admin', 'admin',
                              initial_drop=False))
graph.include(models.Node.registry)
graph.include(models.Relationship.registry)

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('cx')

# Clear all executable circuit nodes:
for class_name in ['LPU', 'Pattern', 'Interface', 'Port',
                   'CircuitModel', 'LeakyIAF', 'AlphaSynapse']:
    r = graph.client.command('delete vertex %s' % class_name)
    logger.info('deleted %s %s nodes' % (r[0], class_name))

logger.info('--- creating LPU, neuron, and synapse model nodes ---')

# Get biological neuron and synapse info:
neuropil_list = graph.Neuropils.query().all()
q = query.QueryWrapper.from_elements(graph, *neuropil_list)
g = q.traverse_owns(max_levels = 1, cls = ['Neuropil', 'Neuron', 'Synapse']).get_as('nx')

# Create LPU nodes:
lpu_name_to_lpu_node = {}   # map LPU name to LPU nodes
for lpu_name in ['BU', 'bu', 'FB', 'EB', 'PB']:
    lpu_name_to_lpu_node[lpu_name] = graph.LPUs.create(name=lpu_name)
    logger.info('created LPU node: %s' % lpu_name)

# Create LeakyIAF and AlphaSynapse nodes for each Neuron and Synapse instance
# node, respectively:
neuropil_family_to_circuit_node = {} # map (neuropil name, neuron family) tuples to circuit nodes
bio_rid_to_exec_node = {}   # map biological neuron/synapse node IDs to executable circuit nodes
for n, data in g.nodes(data=True):

    # Create neuron model node (XXX this doesn't strictly adhere to the
    # NeuroArch data model because no NeuronModel nodes are created):
    if data['class'] == 'Neuron':
        exec_node_class = 'LeakyIAF'
        exec_node = graph.LeakyIAFs.create(name=data['name'])
        bio_rid_to_exec_node[n] = exec_node
        logger.info('created %s node: %s' % (exec_node_class, data['name']))
    elif data['class'] == 'Synapse':
        exec_node_class = 'AlphaSynapse'
        exec_node = graph.AlphaSynapses.create(name=data['name'])
        bio_rid_to_exec_node[n] = exec_node
        logger.info('created %s node: %s' % (exec_node_class, data['name']))
    else:
        logger.info('skipping %s node' % data['class'])
        continue

    # Create CircuitModel node for every neuropil/neuron family combination; we
    # check both neuropil and neuron family because left and right paired
    # neuropils may have the same families of neurons and therefore should be
    # assigned their own circuit nodes, e.g., 
    # NeuropilA_left -> CircuitA -> neurons
    # NeuropilA_right -> CircuitA -> neurons
    neuropil_family_tuple = (data['neuropil'], data['family'])
    if neuropil_family_tuple not in neuropil_family_to_circuit_node:
        neuropil_family_to_circuit_node[neuropil_family_tuple] = \
            graph.CircuitModels.create(name=data['family'])
        logger.info('created CircuitModel node for neuropil %s: %s' %
                    (data['family'], data['neuropil']))

        # Create ownership edge between relevant LPU node and CircuitModel node
        # (take advantage of 'neuropil' attrib inserted during loading of data
        # via ETL or Synapse node creation):
        graph.Owns.create(lpu_name_to_lpu_node[data['neuropil']],
                          neuropil_family_to_circuit_node[neuropil_family_tuple])
        logger.info('connected LPU %s -[Owns]-> CircuitModel %s' % \
                    (data['neuropil'], data['family']))

    # Connect CircuitModel node to LeakyIAF/AlphaSynapse node:
    graph.Owns.create(neuropil_family_to_circuit_node[neuropil_family_tuple],
                      exec_node)
    logger.info('connected CircuitModel %s -[Owns]-> %s %s' % \
                (data['family'], exec_node_class, data['name']))

logger.info('--- creating LPU interfaces, ports, and connections ---')

# Create Interface nodes and attach them to the LPU nodes that own them:
lpu_name_to_int_node = {}
for lpu_name in lpu_name_to_lpu_node:
    lpu_name_to_int_node[lpu_name] = graph.Interfaces.create(name=0)
    graph.Owns.create(lpu_name_to_lpu_node[lpu_name], lpu_name_to_int_node[lpu_name])
    logger.info('connected LPU %s -[Owns]-> Interface %s' % (lpu_name, 0))

# Don't need to look at edge keys even though g is a MultiDiGraph because
# synapses are represented as nodes and hence will not result in multiple edges
# between any two nodes:
ports_to_lpu_name = {}  # maps port node to corresponding LPU name
ports_to_connect = []   # list of port pairs to connect with a pattern
lpu_port_counter = {}   # maps LPU names to counters used to name ports
node_rid_to_out_port = {}   # maps Neuron nodes to output ports created for their LeakyIAF counterparts
for i, j in g.edges():

    # Only look at edges connecting Neuron and Synapse nodes:
    if g.node[i]['class'] not in ['Neuron', 'Synapse'] or \
       g.node[j]['class'] not in ['Neuron', 'Synapse']:
        logger.info('skipping edge between %s -> %s' % (g.node[i]['class'], g.node[j]['class']))
        continue

    # Get names of neuropils that own the connected biological circuit nodes:
    neuropil_id_i = \
        nxtools.in_nodes_has(g, [i], 'class', 'Neuropil')[0]
    neuropil_id_j = \
        nxtools.in_nodes_has(g, [j], 'class', 'Neuropil')[0]
    lpu_name_i = g.node[neuropil_id_i]['name']
    lpu_name_j = g.node[neuropil_id_j]['name']

    # If the connected biological nodes are owned by different neuropils, create
    # Port nodes and attach them to the corresponding executable circuit nodes:
    if lpu_name_i != lpu_name_j:

        # Check if output port has already been created for node with RID i:
        if i in node_rid_to_out_port:
            port_i = node_rid_to_out_port[i]
        else:

            # Create counter used to generate selector names for LPU:
            if lpu_name_i not in lpu_port_counter:
                lpu_port_counter[lpu_name_i] = itertools.count()

            # Create port for node i:
            sel_i = '/%s/out/spk/%i' % (lpu_name_i, lpu_port_counter[lpu_name_i].next())
            port_i = graph.Ports.create(selector=sel_i, port_io='out',
                                        port_type='spike')
            ports_to_lpu_name[port_i] = lpu_name_i
            logger.info('created Port %s, %s, %s' % (sel_i, 'out', 'spike'))

            # Interface nodes must own the new Port nodes:
            graph.Owns.create(lpu_name_to_int_node[lpu_name_i], port_i)
            logger.info('connected LPU Interface %s -[Owns]-> LPU Port %s' % \
                        (lpu_name_i, sel_i))

            # Port transmits data from node i:
            graph.SendsTo.create(bio_rid_to_exec_node[i], port_i)
            logger.info('connected LPU Interface %s -[SendsTo]-> LPU Port %s' % \
                        (lpu_name_i, sel_i))

            # Save port:
            node_rid_to_out_port[i] = port_i

        # Create counter used to generate selector names for LPU:
        if lpu_name_j not in lpu_port_counter:
            lpu_port_counter[lpu_name_j] = itertools.count()

        # Create port for node j:
        sel_j = '/%s/in/spk/%i' % (lpu_name_j, lpu_port_counter[lpu_name_j].next())
        port_j = graph.Ports.create(selector=sel_j, port_io='in',
                                    port_type='spike')
        ports_to_lpu_name[port_j] = lpu_name_j
        logger.info('created Port %s, %s, %s' % (sel_j, 'in', 'gpot'))

        # Interface nodes must own the new Port nodes:
        graph.Owns.create(lpu_name_to_int_node[lpu_name_j], port_j)
        logger.info('connected LPU Interface %s -[Owns]-> LPU Port %s' % \
                    (lpu_name_j, sel_j))

        # Port transmits data to node j:
        graph.SendsTo.create(port_j, bio_rid_to_exec_node[j])
        logger.info('connected LPU Interface %s -[SendsTo]-> LPU Port %s' % \
                    (lpu_name_j, sel_j))

        # Save pair of ports to connect with pattern:
        ports_to_connect.append((port_i, port_j))
    else:

        # If the connected biological nodes are owned by the same neuropil,
        # create a SendsTo edge between the corresponding executable circuit
        # nodes:
        graph.SendsTo.create(bio_rid_to_exec_node[i], bio_rid_to_exec_node[j])
        logger.info('connected %s %s -[SendsTo]-> %s %s' % \
                    (bio_rid_to_exec_node[i].__class__.__name__,
                     bio_rid_to_exec_node[i].name,
                     bio_rid_to_exec_node[j].__class__.__name__,
                     bio_rid_to_exec_node[j].name))

logger.info('--- creating pattern elements ---')

lpu_pair_to_pat_node = {}     # map connected LPU name pairs to patterns
lpu_pair_to_pat_int_node = {} # map connected LPU name pairs to pattern interfaces
for lpu_port_i, lpu_port_j in ports_to_connect:

    # Get LPUs associated with Port instances:
    lpu_name_i = ports_to_lpu_name[lpu_port_i]
    lpu_name_j = ports_to_lpu_name[lpu_port_j]

    # Ignore order of LPUs associated with pair of ports when keeping track of
    # pattern name:
    lpu_pair_set = frozenset((lpu_name_i, lpu_name_j))
    pat_name = '%s-%s' % (lpu_name_i, lpu_name_j)

    # Create Pattern node for every combination of LPU nodes:
    if lpu_pair_set not in lpu_pair_to_pat_node:
        pat = graph.Patterns.create(name=pat_name)
        lpu_pair_to_pat_node[lpu_pair_set] = pat
        logger.info('created Pattern %s' % pat_name)

        # Create two Interface nodes for each Pattern node and connect them to the
        # latter:
        int_0 = graph.Interfaces.create(name=0)
        logger.info('created Interface %s' % 0)
        int_1 = graph.Interfaces.create(name=1)
        logger.info('created Interface %s' % 1)
        graph.Owns.create(pat, int_0)
        logger.info('connected Pattern %s -[Owns]-> Interface %s' % (pat_name, 0))
        graph.Owns.create(pat, int_1)
        logger.info('connected Pattern %s -[Owns]-> Interface %s' % (pat_name, 1))
        lpu_pair_to_pat_int_node[lpu_pair_set] = {lpu_name_i: int_0, lpu_name_j: int_1}
    else:
        pat = lpu_pair_to_pat_node[lpu_pair_set]
        int_0 = lpu_pair_to_pat_int_node[lpu_pair_set][lpu_name_i]
        int_1 = lpu_pair_to_pat_int_node[lpu_pair_set][lpu_name_j]
        logger.info('Pattern %s exists: %s -> %s, %s -> %s' % \
                    (pat_name, lpu_name_i, int_0.name,
                     lpu_name_j, int_1.name))

    # Create new Port nodes corresponding to each of those owned by the two LPU's
    # Interfaces, attach them to Pattern's Interfaces:
    port_io_i = 'in' if lpu_port_i.port_io == 'out' else 'out'
    pat_port_i = graph.Ports.create(selector=lpu_port_i.selector,
                                    port_io=port_io_i,
                                    port_type=lpu_port_i.port_type)
    logger.info('created Port %s, %s, %s' % \
                (lpu_port_i.selector, port_io_i, lpu_port_j.port_type))
    graph.Owns.create(int_0, pat_port_i)
    logger.info('connected Pattern Interface %s -[Owns]-> Pattern Port %s' % \
                (int_0.name, lpu_port_i.selector))

    port_io_j = 'in' if lpu_port_j.port_io == 'out' else 'out'
    pat_port_j = graph.Ports.create(selector=lpu_port_j.selector,
                                    port_io=port_io_j,
                                    port_type=lpu_port_j.port_type)
    logger.info('created Port %s, %s, %s' % \
                (lpu_port_j.selector, port_io_j, lpu_port_j.port_type))
    graph.Owns.create(int_1, pat_port_j)
    logger.info('connected Pattern Interface %s -[Owns]-> Pattern Port %s' % \
                (int_1.name, lpu_port_j.selector))

    # Connect the new Port nodes in the Pattern's two Interface instances together:
    graph.SendsTo.create(pat_port_i, pat_port_j)
    logger.info('connected Pattern Port %s -[SendsTo]-> Pattern Port %s' % \
                (lpu_port_i.selector, lpu_port_j.selector))

    # Connect the Pattern ports to the corresponding LPU ports:
    graph.SendsTo.create(lpu_port_i, pat_port_i)
    logger.info('connected LPU Port %s -[SendsTo]-> Pattern Port %s' % \
                (lpu_port_i.selector, pat_port_i.selector))
    graph.SendsTo.create(pat_port_j, lpu_port_j)
    logger.info('connected Pattern Port %s -[SendsTo]-> LPU Port %s' % \
                (pat_port_j.selector, lpu_port_j.selector))
