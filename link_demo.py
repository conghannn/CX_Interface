#!/usr/bin/env python

"""
Run link demo.
"""

import neurokernel.mpi_relaunch

from pyorient.ogm import Graph, Config
import pyorient.ogm.graph
setattr(pyorient.ogm.graph, 'orientdb_version',
        pyorient.ogm.graph.ServerVersion)

import logging
import sys

import networkx as nx
import numpy as np

import retlam_demo

from neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core
from neurokernel.LPU.LPU import LPU
import neurokernel.pattern as Pattern
from neurokernel.plsel import Selector

import neuroarch.models as models
import neuroarch.nk as nk

from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor

from cx_config import cx_db
from cx_utils import partly_relabel_by_sorted_attr


def cx_component(graph):

	#lpu lists
	lpu_name_list = ['BU', 'bu', 'EB', 'FB', 'PB']
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

		nx.write_gexf(lpu_name_to_g_nk[name], name+'.gexf.gz')



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
	    pat_name_to_pat[name] = Pattern.Pattern.from_graph(nx.DiGraph(pat_name_to_g_nk[name]))

	return lpu_name_to_comp_dict, lpu_name_to_conn_list, pat_name_list, pat_name_to_pat

def get_retina(config):

	num_rings = config['Retina']['rings']
	eulerangles = config['Retina']['eulerangles']
	radius = config['Retina']['radius']

	transform = retlam_demo.AlbersProjectionMap(radius, eulerangles).invmap

	r_hexagon = retlam_demo.r_hx.HexagonArray(num_rings=num_rings, radius=radius, transform=transform)

	retina = retlam_demo.ret.RetinaArray(r_hexagon, config)

	return retina

def get_cx_selectors_name():
	to_list = {} 
	lpu_selectors = {}

	lpu_list = ['BU', 'bu']

	for lpu_name in lpu_list:
		lpu_selectors[lpu_name] = []
		to_list[lpu_name] = []
	
		for i in range(721):
			#sel_j = '/%s/in/gpot/%s' % (lpu_name, i)
			sel_j = '/%s/%s/R1/0' % (lpu_name, i)
			lpu_selectors[lpu_name].append(sel_j)
			to_list[lpu_name].append(sel_j)
	return lpu_selectors, to_list


def link_retina_pat_cx(retina, lpu_selectors, to_list, manager):
	index = 0
	retina_id = retlam_demo.get_retina_id(index)

	retina_all_selectors = retina.get_all_selectors()
	base = 0 
	retina_selectors = []
	from_list = []
	for ret_sel in retina_all_selectors:
		if not ret_sel.endswith('agg/0'):
			_, lpu, ommid, n_name, _  = ret_sel.split('/')
			if int(ommid) != base:
				raise ValueError('selectors order not matched')
			from_list.append(ret_sel)
			retina_selectors.append(ret_sel)
			base += 1

	for lpu_name in ['BU', 'bu']:
		pattern = Pattern.Pattern.from_concat(','.join(retina_selectors),
		                          ','.join(lpu_selectors[lpu_name]),
		                          from_sel=','.join(from_list),
		                          to_sel=','.join(to_list[lpu_name]),
		                          spike_sel=','.join(from_list + to_list[lpu_name]))


		nx.write_gexf(pattern.to_graph(), retina_id+'_'+lpu_name+'.gexf.gz', prettyprint=True)
		
		manager.connect(retina_id, lpu_name, pattern)



def main():

	logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
	                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
	logger = logging.getLogger('cx')



	sys.setrecursionlimit(10000)
	### Graph ###
	graph = Graph(Config.from_url(cx_db, 'admin', 'admin',
	                              initial_drop=False))
	graph.include(models.Node.registry)
	graph.include(models.Relationship.registry)


	### Retina ###
	config=retlam_demo.ConfigReader('retlam_default.cfg','../template_spec.cfg').conf

	retina = get_retina(config)
	

	##### Configuration  ######
	logger = setup_logger(screen=True)
	lpu_selectors, to_list = get_cx_selectors_name()
	lpu_name_to_comp_dict, lpu_name_to_conn_list, pat_name_list, pat_name_to_pat = cx_component(graph)
	

	man = core.Manager()

	dt = 1e-4
	dur = 0.2
	steps = int(dur/dt)
	debug = True

	lpu_name_list = ['BU', 'bu', 'EB', 'FB', 'PB']
	for name in lpu_name_list:
		input_processor = []
		output_processor = [FileOutputProcessor([('spike_state', None), ('V',None), ('g',None), ('I', None)], '{}_output.h5'.format(name), sample_interval = 1)]

		man.add(LPU, name, dt, lpu_name_to_comp_dict[name],
	            lpu_name_to_conn_list[name],
	            input_processors = input_processor,
	            output_processors = output_processor,
	            device=0,
	            debug=debug, time_sync=False)


	retlam_demo.add_retina_LPU(config, 0, retina, man)
	logger.info('add retina lpu')

	for name in pat_name_list:
	    id_0, id_1 = name.split('-')
	    man.connect(id_0, id_1, pat_name_to_pat[name][0], pat_name_to_pat[name][1].index('0'), pat_name_to_pat[name][1].index('1'))

	logger.info('link lpus among cx lpus')    

	link_retina_pat_cx(retina, lpu_selectors, to_list, man)
	logger.info('link retina and cx lpu')
	
	man.spawn()
	man.start(steps)
	man.wait()
	
if __name__ == '__main__':
    main()