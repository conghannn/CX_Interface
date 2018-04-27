#!/usr/bin/env python

import itertools
import logging
import sys

from pyorient.ogm import Graph, Config

import neuroarch.models as models
import neuroarch.query as query
import arbor_funcs

from cx_config import cx_db

graph = Graph(Config.from_url(cx_db, 'admin', 'admin',
                              initial_drop=False))
graph.include(models.Node.registry)
graph.include(models.Relationship.registry)

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('cx')

# Get arborizations for all neuropils:
neuropil_list = graph.Neuropils.query().all()
q = query.QueryWrapper.from_elements(graph, *neuropil_list)
df_neurons, _ = q.owns(1, cls = 'Neuron').get_as('df')
df_arbors, _ = q.owns(2, cls = 'ArborizationData').get_as('df')

# Convert lists of regions in arborization data to sets:
for rid in df_arbors.index:
    df_arbors.loc[rid]['regions'] = \
        set([tuple(r) if isinstance(r, list) \
             else r for r in df_arbors.loc[rid]['regions']])
    df_arbors.loc[rid]['neurite'] = set(df_arbors.loc[rid]['neurite'])

# Find neurons with overlapping arborizations that have opposite polarities:
nodes_to_connect = []
for rid0, rid1 in itertools.combinations(df_neurons.index, 2):
    n0 = df_neurons.ix[rid0]['name']
    n1 = df_neurons.ix[rid1]['name']
    arbors0 = df_arbors[df_arbors['neuron'] == n0]
    arbors1 = df_arbors[df_arbors['neuron'] == n1]
    d = set()
    for a0, a1 in itertools.product(arbors0.to_dict('records'),
                                    arbors1.to_dict('records')):
        d.update(arbor_funcs.check_overlap(a0, a1))

        # No need to check more arborizations if connection in both directions
        # are already implied:
        if d == {'r', 'l'}:
            break
    if 'r' in d:
        nodes_to_connect.append((rid0, rid1))
        logger.info('connection discovered: %s -> %s' % (rid0, rid1))
    if 'l' in d:
        nodes_to_connect.append((rid1, rid0))
        logger.info('connection discovered: %s <- %s' % (rid0, rid1))

# Create a new synapse node for every connection and connect the input and
# output nodes of the connection to the synapse node:
for nc in nodes_to_connect:

    # Create a new synapse node for every connection:
    name = df_neurons.ix[nc[0]]['name']+'->'+df_neurons.ix[nc[1]]['name']
    syn = graph.Synapses.create(name=name)
    logger.info('created Synapse node: %s' % name)

    # Save family and neuropil of postsynaptic neuron in the synapse node:
    syn.update(family=df_neurons.ix[nc[1]]['family'],
               neuropil=df_neurons.ix[nc[1]]['neuropil'])

    from_neuropil = graph.get_element(nc[0]).in_(models.Owns)[0]
    to_neuropil = graph.get_element(nc[1]).in_(models.Owns)[0]
    if from_neuropil != to_neuropil:
        logger.info('elements in different neuropils: %s, %s' % \
                    (from_neuropil.name, to_neuropil.name))
    else:
        logger.info('elements in the same neuropil: %s' % from_neuropil.name)

    # Connect the neurons associated with the synapse node (XXX this
    # relationship isn't specified in the NeuroArch data model):
    graph.SendsTo.create(graph.get_element(nc[0]), syn)
    logger.info('connected %s %s -[SendsTo]-> Synapse %s' % \
                (graph.get_element(nc[0]).__class__.__name__,
                 graph.get_element(nc[0]).name,
                 syn.name))
    graph.SendsTo.create(syn, graph.get_element(nc[1]))
    logger.info('connected Synapse %s -[SendsTo]-> %s %s' % \
                (syn.name,
                 graph.get_element(nc[1]).__class__.__name__,
                 graph.get_element(nc[1]).name))

    # Set the postsynaptic neuropil to own the synapse node:
    graph.Owns.create(to_neuropil, syn)
    logger.info('connected Neuropil %s -[Owns]-> Synapse %s' % \
                (to_neuropil.name, syn.name))
