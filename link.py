#!/usr/bin/env python

"""
Run link .
"""
import itertools
import logging
import sys

from pyorient.ogm import Graph, Config

import networkx as nx
import numpy as np

import neuroarch.conv as conv
import neuroarch.models as models
import neuroarch.query as query
import neuroarch.nxtools as nxtools

from cx_config import cx_db

import retlam_demo

from neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core
from neurokernel.LPU.LPU import LPU
import neurokernel.pattern as Pattern
from neurokernel.plsel import Selector

import neuroarch.models as models
import neuroarch.nk as nk
import math


from cx_utils import partly_relabel_by_sorted_attr


def get_neuron_loc_Bb(lpu, square = 1, disc = 0, x_max = 0.778):
	
	'''
	### retina.hex_array._loc:   x_max = 0.866, y_max = 0.875 ###

	mapping_type: rectangle -> [square] -> [disc]
	1. rectangle -> square : x_max = y_max= 0.778 (between hex_array and square, there is a margin that is a radiu of small circle )
	2. rectangle -> square -> disc, x_max = y_max = 0.875
	3. rectangle -> square -> disc, x_max = y_max = 0.8125, a margin
	4. rectangle -> disc, x_max = 0.865, y_max = x_max * 0.8
	5. rectangle -> disc. y_max = 0.875, x_max = y_max / 0.8
	'''
	shape = (8, 10)
	N_y, N_x = shape

	if square == 1:
		y_max = x_max
	else:
		y_max = x_max * (N_y/N_x)

	n_x_offsets = np.linspace(-x_max, x_max, N_x)
	n_y_offsets = np.linspace(y_max, -y_max, N_y)

	neuron_loc_square = []
	for i in range(80):
		y_idx = i % 8
		if lpu == 'BU':
			x_idx = i / 8
		if lpu == 'bu':
			x_idx = 9 - (i / 8)
		#neuron_loc.append((x_idx, y_idx))
		neuron_loc_square.append((n_x_offsets[x_idx], n_y_offsets[y_idx]))

	if disc == 1:
		neuron_loc_disc = []
		for item in neuron_loc_square:
			x = item[0]
			y = item[1]
			u = x * math.sqrt(1 - (y**2)/2.0)
			v = y * math.sqrt(1 - (x**2)/2.0)
			neuron_loc_disc.append((u, v))
		return np.array(neuron_loc_disc)
	else:
		return np.array(neuron_loc_square)


def get_map_Bb(lpu, retina, k_num = 9):
	

	sigma = 0.05

	hexarray = retina.hex_array
	location = hexarray._loc
	mat = np.zeros(shape = (80, 721)).astype(np.float32)
	

	neuron_loc = get_neuron_loc_Bb(lpu, square = 1, disc = 1, x_max = 0.8125)

	for i in range(80):
		for j in range(721):
			neuron_point = neuron_loc[i]
			hex_point = location[j]
			x = neuron_point[0] - hex_point[0]
			y = neuron_point[1] - hex_point[1]
			mat[i][j] = (1.0/(1*np.pi*(sigma**2)))*np.exp(-(1.0/(2*(sigma**2)))*(x**2+y**2))

	neuron_map_retina = {}
	for i in range(80):
		neuron_map_retina[i] = []
		index_group = mat[i].argsort()[-k_num:][::-1]
		sum = mat[i][index_group].sum()
		for n in index_group:
			neuron_map_retina[i].append((n, mat[i][n]/sum))

	return neuron_map_retina

def get_map_p(retina):

	shape = (8, 10)

	hexarray = retina.hex_array
	test = hexarray._loc

	x_max = 1.0
	width = 2.0/18

	n_x_offsets = np.linspace(-x_max, x_max, 18)

	neuron_loc = []
	for i in range(18):
		neuron_loc.append(n_x_offsets[i])
	neuron_loc = np.array(neuron_loc)

	neuron_map_retina = {}
	for i in range(18):
		neuron_map_retina[i] = []
		neuron_point = neuron_loc[i]
		sum = 0
		for j in range(721):
			hex_point = test[j]
			x = neuron_point - hex_point[0]
			if x > -width/2.0 and x <= width/2.0:
				neuron_map_retina[i].append((j, abs(x)))
				sum += abs(x)
		for k, n in enumerate(neuron_map_retina[i]):
			neuron_map_retina[i][k] = (n[0], n[1]/sum)

	return neuron_map_retina


def get_map(retina):
	
	neuropil_map_retina = {}
	neuropil_map_retina['BU'] = get_map_Bb('BU', retina, 9)
	neuropil_map_retina['bu'] = get_map_Bb('bu', retina, 9)
	neuropil_map_retina['PB'] = get_map_p(retina)
	return neuropil_map_retina

def alpha_synapse_params(lpu, weight = 1):
    """
    Generate AlphaSynapse params.
    """
    s = 0.04
    k = 1000
    if lpu == 'BU' or lpu == 'bu':
        return {'conductance': True,
                'ad': 0.16*1000,
                'ar': 1.1*100,
                'gmax': weight * 0.01 * s,
                'reverse': -0.065 * k}
    elif lpu == 'EB':
        return {'conductance': True,
                'ad': 0.16*1000,
                'ar': 1.1*100,
                'gmax': 0.01,
                'reverse': 0.065* k}
    elif lpu == 'FB':
        return {'conductance': True,
                'ad': 0.16*1000,
                'ar': 1.1*100,
                'gmax': 0.01,
                'reverse': 0.065* k}
    elif lpu == 'PB':
        return {'conductance': True,
                'ad': 0.19*1000,
                'ar': 1.1*100,
                'gmax': weight * 0.002,
                'reverse': 0.065* k}
    else:
        raise ValueError('unrecognized LPU name')

def re_update_syn(graph):
	lpu_name_list = ['BU', 'bu']
	for lpu_name in lpu_name_list:
		lpu_cx_node = graph.LPUs.query(name=lpu_name).one()
		lpu_cx_to_query_syn = lpu_cx_node.owns(2, cls = ['AlphaSynapse'])
		if len(lpu_cx_to_query_syn.nodes_as_objs) != 720:
			raise ValueError('number of syn is wrong')

		for n in lpu_cx_to_query_syn.nodes_as_objs:
			n.update(**alpha_synapse_params(lpu_name, 1))
		lpu_cx_to_query_syn.execute(True, True)





def lpu_region_to_number():
	lpu_region_to_vision_region = {}
	BU_to_vision_region = {}
	bu_to_vision_region = {}
	PB_to_vision_region = {}
	for i in range(1, 81):
		BU_to_vision_region['L%s' % i] = i - 1
		bu_to_vision_region['R%s' % i] = i - 1

	tmp = np.concatenate((np.arange(1, 10)[::-1], np.arange(1, 10)))
	for i in range(9):
	    PB_to_vision_region['L%s' % tmp[i]] = i
	for i in range(9, 18):
	    PB_to_vision_region['R%s' % tmp[i]] = i
	lpu_region_to_vision_region['BU'] = BU_to_vision_region
	lpu_region_to_vision_region['bu'] = bu_to_vision_region
	lpu_region_to_vision_region['PB'] = PB_to_vision_region

	return lpu_region_to_vision_region

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


def re_design_cx(graph, retina, logger):

	neuropil_map_retina = get_map(retina)
	lpu_region_to_vision_region = lpu_region_to_number()
	#to_list = {}
	#lpu_selectors = {}

	lpu_list = ['BU', 'bu']
	for lpu_name in lpu_list:

		#cx lpu
		# to check the region of each node in cx lpu
		r = graph.client.query(("select * from "
	            "(select name,out()[@class='ArborizationData'][neuropil='{lpu_name}'].regions "
	            "as regions from Neuron where name in "
	            "(select expand(out()[@class='CircuitModel'].out()"
	            "[@class='LeakyIAF'][extern=true].name) from LPU "
	            "where name = '{lpu_name}' limit -1)"
	            "limit -1) where regions is not null").format(lpu_name=lpu_name))

		df = conv.pd.as_pandas(r)[0]

		lpu_cx_node = graph.LPUs.query(name=lpu_name).one()

		#get the interface and circuit node of cx lpu and check the number of ports used
		cx_interface = 0     
		cx_circuit = 0
		for n in lpu_cx_node.owns(1).nodes_as_objs:
			if isinstance(n, models.Interface):
				cx_interface = n
			if isinstance(n, models.CircuitModel):
				cx_circuit = n
		#port_num = len(cx_interface.owns(1).nodes_as_objs)

		if not isinstance(cx_interface, models.Interface) or not isinstance(cx_circuit, models.CircuitModel):
			raise ValueError('interface or circuit not found')

		lpu_index_to_ports_name = {}
		lpu_index_to_ports = {}

		#lpu_selectors[lpu_name] = []
		#to_list[lpu_name] = []

		for i in range(721):
			#sel_j = '/%s/in/gpot/%s' % (lpu_name, i)
			sel_j = '/%s/%s/R1/0' % (lpu_name, i)
			port_j = graph.Ports.create(selector=sel_j, port_io='in', port_type='spike')
			logger.info('created Port %s, %s, %s' % (sel_j, 'in', 'spike'))

			graph.Owns.create(cx_interface, port_j)
			logger.info('connected LPU Port %s -[Owns]-> LPU Interface %s' % (sel_j, lpu_name))

			lpu_index_to_ports_name[i] = sel_j
			lpu_index_to_ports[i] = port_j

			#lpu_selectors[lpu_name].append(sel_j)
			#to_list[lpu_name].append(sel_j)

		# create syn in cx lpu and in_port
		lpu_cx_to_query = lpu_cx_node.owns(2, cls = ['LeakyIAF'])
		for i, n in enumerate(lpu_cx_to_query.nodes_as_objs):
			if isinstance(n, models.LeakyIAF):
				regions = df[df['name'] == n.name]['regions'][0][0]
				region_num = lpu_region_to_vision_region[lpu_name][regions]

				port_mapping = neuropil_map_retina[lpu_name][region_num]
				print port_mapping

			for item in port_mapping:
				print item
				port_id = item[0]
				weight = item[1]

				name = 'port%s -> neuron%s in %s' % (str(port_id),  str(i), lpu_name)

				exec_node_class = 'AlphaSynapse'
				exec_node = graph.AlphaSynapses.create(name=name)
				logger.info('created %s node: %s' % (exec_node_class, exec_node.name))

				exec_node.update(**alpha_synapse_params(lpu_name, weight))
				logger.info('update %s node: %s Weight %s' % (exec_node_class, exec_node.name, str(weight)))

				graph.Owns.create(cx_circuit, exec_node)
				logger.info('connected CircuitModel %s -[Owns]-> %s %s' % (cx_circuit.name, exec_node_class, exec_node.name))

				graph.SendsTo.create(lpu_index_to_ports[port_id], exec_node)
				logger.info('connected LPU Port %s -[SendsTo]-> syn %s' % (lpu_index_to_ports_name[port_id], exec_node.name))

				graph.SendsTo.create(exec_node, n)
				logger.info('connected syn %s -[SendsTo]-> %s cx_lpu_exec_node' % (exec_node.name, n.name))

		port_num_print = len(cx_interface.owns(1).nodes_as_objs)
		syn_num_print = len(cx_circuit.owns(1, cls = 'AlphaSynapse').nodes_as_objs)

		logger.info('ports number %s and syn number %s in LPU %s' % (port_num_print, syn_num_print, lpu_name))

		lpu_cx_to_query_syn = lpu_cx_node.owns(2, cls = ['AlphaSynapse'])
		lpu_cx_to_query_syn.execute(True, True)

	#return to_list, lpu_selectors

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



	
def main():

	RecurrentLimit = 10000

	sys.setrecursionlimit(RecurrentLimit)

	logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
	                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
	logger = logging.getLogger('cx')


	### Graph ###
	graph = Graph(Config.from_url(cx_db, 'admin', 'admin',
	                              initial_drop=False))
	graph.include(models.Node.registry)
	graph.include(models.Relationship.registry)


	### Retina ###
	config=retlam_demo.ConfigReader('retlam_default.cfg','../template_spec.cfg').conf

	retina = get_retina(config)

	### Core ###
	
	re_design_cx(graph, retina, logger)



	
if __name__ == '__main__':
    main()






