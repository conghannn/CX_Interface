#!/usr/bin/env python

"""
"""

import itertools
import logging
import sys

import h5py
import networkx as nx
import numpy as np
from pyorient.ogm import Graph, Config

import neuroarch.models as models
import neuroarch.nk as nk
import neuroarch.conv as conv
from neurokernel.LPU.LPU import LPU

from cx_config import cx_db
from cx_utils import partly_relabel_by_sorted_attr

graph = Graph(Config.from_url(cx_db, 'admin', 'admin',
                              initial_drop=False))
graph.include(models.Node.registry)
graph.include(models.Relationship.registry)

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('cx')

# LPUs receiving input:
lpu_name_list = ['BU', 'bu', 'PB']

# Load LPU info:
lpu_name_to_node = {}      # LPU name -> pyorient node
lpu_name_to_g_na = {}      # LPU name -> NeuroArch-compatible graph
lpu_name_to_g_nk_orig = {} # LPU name -> Neurokernel-compatible graph
lpu_name_to_g_nk = {}      # LPU name -> Neurokernel-compatible graph with int IDs
#lpu_name_to_n_dict = {}    # LPU name -> n_dict data struct
#lpu_name_to_s_dict = {}    # LPU name -> s_dict data struct
lpu_name_to_comp_dict = {} # LPU name -> comp_dict data struct
lpu_name_to_conn_list = {} # LPU name -> conn_list data struct

for name in lpu_name_list:
    logger.info('extracting data for LPU %s' % name)
    lpu_name_to_node[name] = graph.LPUs.query(name=name).one()
    lpu_name_to_g_na[name] = lpu_name_to_node[name].traverse_owns(max_levels = 2).get_as('nx')
    lpu_name_to_g_nk_orig[name] = nk.na_lpu_to_nk_new(lpu_name_to_g_na[name])
    lpu_name_to_g_nk[name] = nx.convert_node_labels_to_integers(lpu_name_to_g_nk_orig[name], ordering = 'sorted')
    lpu_name_to_g_nk[name] = \
        partly_relabel_by_sorted_attr(lpu_name_to_g_nk[name], 'model', ['LeakyIAF'], 'name')
    #lpu_name_to_n_dict[name], lpu_name_to_s_dict[name] = LPU.graph_to_dicts(lpu_name_to_g_nk[name])
    lpu_name_to_comp_dict[name], lpu_name_to_conn_list[name] = LPU.graph_to_dicts(lpu_name_to_g_nk[name])

# These maps from region identifier numbers to receptive field coordinates are
# hypothetical!
lpu_name_to_rf_map = {}

# Map between microglomerulus region to receptive field in BU/bu:
logger.info('setting up maps between regions and RFs')
tmp = np.arange(1, 81).reshape((8, 10), order='F') # [[1, 2, 3, ..], [9, 10, 11..]]
BU_region_to_rf = {}
for i, j in itertools.product(range(tmp.shape[0]), range(tmp.shape[1])):
    BU_region_to_rf['L%s' % tmp[i, j]] = (i, j)
lpu_name_to_rf_map['BU'] = BU_region_to_rf

tmp = np.arange(1, 81).reshape((8, 10), order='F')[:, ::-1] # [[73, 74, 75, ..], [65, 66, 67, ..]]
bu_region_to_rf = {}
for i, j in itertools.product(range(tmp.shape[0]), range(tmp.shape[1])):
    bu_region_to_rf['R%s' % tmp[i, j]] = (i, j)
lpu_name_to_rf_map['bu'] = bu_region_to_rf

# Map between glomerulus region to receptive field in PB:
PB_region_to_rf = {}
tmp = np.concatenate((np.arange(1, 10)[::-1],
                      np.arange(1, 10)))
for i in range(9):
    PB_region_to_rf['L%s' % tmp[i]] = i
for i in range(9, 18):
    PB_region_to_rf['R%s' % tmp[i]] = i
lpu_name_to_rf_map['PB'] = PB_region_to_rf

for lpu_name in lpu_name_list:

    # Load input signal array for current LPU; the input array should contain 
    # one 1D signal for each region receiving input:
    logger.info('loading %s input data' % lpu_name)
    with h5py.File('%s_input_pre.h5' % lpu_name, 'r') as f:
        data_pre = f['/array'][:]

    # Create table of neuron names and arborization regions per neuron:
    r = graph.client.query(("select * from "
        "(select name, out()[@class='ArborizationData'][neuropil='{lpu_name}'].regions "
        "as regions from Neuron where name in "
        "(select expand(out()[@class='CircuitModel'].out()"
        "[@class='LeakyIAF'][extern=true].name) from LPU "
        "where name = '{lpu_name}' limit -1)"
        "limit -1) where regions is not null").format(lpu_name=lpu_name))
    df = conv.pd.as_pandas(r)[0]

    # The output array must contain as many 1D input signals as there are
    # neurons in the current LPU receiving input:
    data = np.zeros((data_pre.shape[0], len(df['name'].unique())))
    logger.info('creating output array with shape %s' % str(data.shape))

    #for i, name in enumerate(lpu_name_to_n_dict[lpu_name]['LeakyIAF']['name']):
    for i, name in enumerate(lpu_name_to_comp_dict[lpu_name]['LeakyIAF']['name']):
        # Get the arborization regions within the current LPU:
        regions = df[df['name'] == name]['regions'][0]

        # Find the indices of the RFs corresponding to the regions associated with
        # neuron i:
        tmp = [lpu_name_to_rf_map[lpu_name][region] for region in regions]
        idx_list = [(idx,) if not isinstance(idx, tuple) else idx for idx in tmp]

        # Get the maximum RF response over all of the regions associated with a
        # specific neuron:
        print lpu_name, regions
        data_pre_selected = np.array([data_pre[(slice(None, None),)+idx] for idx in idx_list])
        data[:, i] = np.array([np.max(col) for col in data_pre_selected.T])

    with h5py.File('%s_input.h5' % lpu_name, 'w') as f:
        f.create_dataset('/I/uids', data = np.array(lpu_name_to_comp_dict[lpu_name]['LeakyIAF']['id']))
        #f.create_dataset('/array', data=data)
        f.create_dataset('/I/data', data=data)
