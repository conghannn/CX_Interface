#!/usr/bin/env python

"""
Run CX demo.
"""

import neurokernel.mpi_relaunch

import networkx as nx
import numpy as np
from pyorient.ogm import Graph, Config
import pyorient.ogm.graph

# Required to handle dill's inability to serialize namedtuple class generator:
setattr(pyorient.ogm.graph, 'orientdb_version',
        pyorient.ogm.graph.ServerVersion)

from neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core
from neurokernel.LPU.LPU import LPU
import neurokernel.pattern as pattern
from neurokernel.plsel import Selector

import neuroarch.models as models
import neuroarch.nk as nk

from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor

from cx_config import cx_db
from cx_utils import partly_relabel_by_sorted_attr

graph = Graph(Config.from_url(cx_db, 'admin', 'admin',
                              initial_drop=False))
models.create_efficiently(graph, models.Node.registry)
models.create_efficiently(graph, models.Relationship.registry)

logger = setup_logger(screen=True)

lpu_name_list = ['BU', 'bu', 'EB', 'FB', 'PB']


# LPUs:
lpu_name_to_node = {}      # LPU name -> pyorient LPU node
lpu_name_to_g_na = {}      # LPU name -> NeuroArch-compatible graph
lpu_name_to_g_nk_orig = {} # LPU name -> Neurokernel-compatible graph
lpu_name_to_g_nk = {}      # LPU name -> Neurokernel-compatible graph with int IDs
#lpu_name_to_n_dict = {}    # LPU name -> n_dict data struct
#lpu_name_to_s_dict = {}    # LPU name -> s_dict data struct
lpu_name_to_comp_dict = {} # LPU name -> comp_dict data struct
lpu_name_to_conn_list = {} # LPU name -> conn_list data struct

for name in lpu_name_list:
    lpu_name_to_node[name] = graph.LPUs.query(name=name).one()
    lpu_name_to_g_na[name] = lpu_name_to_node[name].traverse_owns(max_levels = 2).get_as('nx')
    lpu_name_to_g_nk_orig[name] = nk.na_lpu_to_nk_new(lpu_name_to_g_na[name])
    lpu_name_to_g_nk[name] = nx.convert_node_labels_to_integers(lpu_name_to_g_nk_orig[name], ordering = 'sorted')
    lpu_name_to_g_nk[name] = \
        partly_relabel_by_sorted_attr(lpu_name_to_g_nk[name], 'model', ['LeakyIAF'], 'name')
    #lpu_name_to_n_dict[name], lpu_name_to_s_dict[name] = LPU.graph_to_dicts(lpu_name_to_g_nk[name])
    lpu_name_to_comp_dict[name], lpu_name_to_conn_list[name] = LPU.graph_to_dicts(lpu_name_to_g_nk[name])

# Patterns:
pat_name_list = [n.name for n in graph.Patterns.query().all()]

pat_name_to_node = {}     # LPU pair -> pyorient Pattern node
pat_name_to_g_na = {}     # LPU pair -> NeuroArch-compatible graph
pat_name_to_g_nk = {}     # LPU pair -> Neurokernel-compatible graph
pat_name_to_pat = {}      # LPU pair -> Pattern class instance

for name in pat_name_list:
    pat_name_to_node[name] = graph.Patterns.query(name=name).one()
    pat_name_to_g_na[name] = pat_name_to_node[name].traverse_owns(max_levels = 2).get_as('nx')
    pat_name_to_g_nk[name] = nk.na_pat_to_nk(pat_name_to_g_na[name])
    pat_name_to_pat[name] = pattern.Pattern.from_graph(nx.DiGraph(pat_name_to_g_nk[name]))

man = core.Manager()

dt = 1e-4
dur = 0.2
steps = int(dur/dt)

debug = False
for name in lpu_name_list:
    if name in ['BU', 'bu', 'PB']:
        #in_file_name = '%s_input.h5' % name
        input_processor = [FileInputProcessor('{}_input.h5'.format(name))]
    else:
        #in_file_name = None
        input_processor = []
    #out_file_name = '%s_output.h5' % name
    output_processor = [FileOutputProcessor([('spike_state', None)], '{}_output.h5'.format(name), sample_interval = 1)]

    # Since the LPUs are relatively small, they can all use the same GPU:
    # man.add(LPU, name, dt, lpu_name_to_n_dict[name],
    #         lpu_name_to_s_dict[name],
    #         input_file=in_file_name,
    #         output_file=out_file_name,
    #         device=0,
    #         debug=debug, time_sync=False)
    man.add(LPU, name, dt, lpu_name_to_comp_dict[name],
            lpu_name_to_conn_list[name],
            input_processors = input_processor,
            output_processors = output_processor,
            device=0,
            debug=debug, time_sync=False)

check_compatibility = False
if check_compatibility:
    lpu_name_to_sel_in_gpot = {}
    lpu_name_to_sel_in_spike = {}
    lpu_name_to_sel_out_gpot = {}
    lpu_name_to_sel_out_spike = {}
    lpu_name_to_sel_in = {}
    lpu_name_to_sel_out = {}
    lpu_name_to_sel_gpot = {}
    lpu_name_to_sel_spike = {}
    lpu_name_to_sel = {}

    for name in lpu_name_list:
        n_dict = lpu_name_to_n_dict[name]
        lpu_name_to_sel_in_gpot[name] = \
            Selector(LPU.extract_in_gpot(n_dict))
        lpu_name_to_sel_in_spike[name] = \
            Selector(LPU.extract_in_spk(n_dict))
        lpu_name_to_sel_out_gpot[name] = \
            Selector(LPU.extract_out_gpot(n_dict))
        lpu_name_to_sel_out_spike[name] = \
            Selector(LPU.extract_out_spk(n_dict))
        lpu_name_to_sel_in[name] = \
            Selector.union(lpu_name_to_sel_in_gpot[name], lpu_name_to_sel_in_spike[name])
        lpu_name_to_sel_out[name] = \
            Selector.union(lpu_name_to_sel_out_gpot[name], lpu_name_to_sel_out_spike[name])
        lpu_name_to_sel_gpot[name] = \
            Selector.union(lpu_name_to_sel_in_gpot[name], lpu_name_to_sel_out_gpot[name])
        lpu_name_to_sel_spike[name] = \
            Selector.union(lpu_name_to_sel_in_spike[name], lpu_name_to_sel_out_spike[name])
        lpu_name_to_sel[name] = Selector.union(lpu_name_to_sel_in[name], lpu_name_to_sel_out[name])


    lpu_name_to_int = {}
    for name in lpu_name_list:
        lpu_name_to_int[name] = \
                pattern.Interface.from_selectors(lpu_name_to_sel[name],
                                                 lpu_name_to_sel_in[name],
                                                 lpu_name_to_sel_out[name],
                                                 lpu_name_to_sel_spike[name],
                                                 lpu_name_to_sel_gpot[name],
                                                 lpu_name_to_sel[name])

    for name in pat_name_list:
        id_0, id_1 = name.split('-')

        assert lpu_name_to_int[id_0].is_compatible(0, pat_name_to_pat[name].interface, 0, True)
        assert lpu_name_to_int[id_1].is_compatible(0, pat_name_to_pat[name].interface, 1, True)

for name in pat_name_list:
    id_0, id_1 = name.split('-')
    man.connect(id_0, id_1, pat_name_to_pat[name][0], pat_name_to_pat[name][1].index('0'), pat_name_to_pat[name][1].index('1'))

man.spawn()
man.start(steps)
man.wait()
