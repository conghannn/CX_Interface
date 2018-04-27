#!/usr/bin/env python

"""
Visualize LPU outputs.
"""

import argparse

import networkx as nx
import numpy as np
import matplotlib as mpl
mpl.use('agg')
from neurokernel.tools.logging import setup_logger
import neuroarch.models as models
import neuroarch.nk as nk
import neuroarch.conv as conv
from neurokernel.LPU.LPU import LPU
import neurokernel.LPU.utils.visualizer as vis

nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                       'true':True, 'True':True}

from pyorient.ogm import Graph, Config

from cx_config import cx_db
from cx_utils import partly_relabel_by_sorted_attr

graph = Graph(Config.from_url(cx_db, 'admin', 'admin',
                              initial_drop=False))

models.create_efficiently(graph, models.Node.registry)
models.create_efficiently(graph, models.Relationship.registry)

import argparse

import h5py


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



parser = argparse.ArgumentParser()
parser.add_argument('-d', default='l2r', type=str,
                    help='Direction [l2r, r2l]')
args = parser.parse_args()

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
    lpu_name_to_comp_dict[name], lpu_name_to_conn_list[name] = LPU.graph_to_dicts(lpu_name_to_g_nk[name])


lpu_name_to_family_map = {}
for name in ['EB', 'PB']:
    r = graph.client.query(("select name, family from Neuron where name in "
                "(select expand(out()[@class='CircuitModel'].out()[@class='LeakyIAF'][extern=true].name)"
                           "from LPU where name = '{name}' limit -1)").format(name=name))
    df = conv.pd.as_pandas(r)[0]
    lpu_name_to_family_map[name] = {r['name']:r['family'] for r in df.to_dict('record')}


sort_list = {}

############# BU and bu ####################
lpu_region_to_vision_region = lpu_region_to_number()
for name in ['BU', 'bu']:
	filename = '%s_output.h5' %name
	f = h5py.File(filename, 'r')
	uids = f['spike_state']['uids'][:]
	data_T = f['spike_state']['data'][:].T

	sort_list_dict = {}
	sort_list_list = []

  
	r = graph.client.query(("select * from "
	      "(select name, out()[@class='ArborizationData'][neuropil='{lpu_name}'].regions "
	      "as regions from Neuron where name in "
	      "(select expand(out()[@class='CircuitModel'].out()"
	      "[@class='LeakyIAF'][extern=true].name) from LPU "
	      "where name = '{lpu_name}' limit -1)"
	      "limit -1) where regions is not null").format(lpu_name=name))
	df = conv.pd.as_pandas(r)[0]
  
	for k, n in lpu_name_to_g_nk[name].node.items():
		if n['class'] == 'LeakyIAF':
			neuron_name = n['name']
			regions = df[df['name'] == neuron_name]['regions'][0][0]
			region_num = lpu_region_to_vision_region[name][regions]
			sort_list_dict[region_num] = int(k)

	id_positions = []
	for i in range(len(df)):
		sort_list_list.append(sort_list_dict[i])    

		position = list(uids).index(sort_list_dict[i])
		id_positions.append(position)

	sort_list[name] = sort_list_list
	data_out = data_T[id_positions, :].T

	with h5py.File('%s_output_pick.h5' % name, 'w') as f:
		f.create_dataset('/array', data = data_out)

########################################## EB ################################################

name = 'EB'
filename = '%s_output.h5' %name
f = h5py.File(filename, 'r')
uids = f['spike_state']['uids'][:]
data_T = f['spike_state']['data'][:].T

sort_list_dict = {}
sort_list_list = []

r = graph.client.query(("select * from "
      "(select name, out()[@class='ArborizationData'][neuropil='{lpu_name}'].regions "
      "as regions from Neuron where name in "
      "(select expand(out()[@class='CircuitModel'].out()"
      "[@class='LeakyIAF'][extern=true].name) from LPU "
      "where name = '{lpu_name}' limit -1)"
      "limit -1) where regions is not null").format(lpu_name=name))
df = conv.pd.as_pandas(r)[0]

for k, n in lpu_name_to_g_nk[name].node.items():
	if n['class'] == 'LeakyIAF' and lpu_name_to_family_map['EB'][n['name']] == 'EB-LAL-PB':
		neuron_name = n['name']
		if neuron_name == 'EB/([L8,L6],[P,M],[1-4])/s-EB/(L7,[P,M],[1-4])/b-lal/RVG/b-PB/L8/b':
			sort_list_dict[0] = int(k)
		elif neuron_name == 'EB/([L5,L7],[P,M],[1-4])/s-EB/(L6,[P,M],[1-4])/b-LAL/LVG/b-PB/R2/b':
			sort_list_dict[1] = int(k)
		elif neuron_name == 'EB/([L6,L4],[P,M],[1-4])/s-EB/(L5,[P,M],[1-4])/b-lal/RDG/b-PB/L7/b':
			sort_list_dict[2] = int(k)
		elif neuron_name == 'EB/([L3,L5],[P,M],[1-4])/s-EB/(L4,[P,M],[1-4])/b-LAL/LDG/b-PB/R3/b':
			sort_list_dict[3] = int(k)
		elif neuron_name == 'EB/([L4,L2],[P,M],[1-4])/s-EB/(L3,[P,M],[1-4])/b-lal/RVG/b-PB/L6/b':
			sort_list_dict[4] = int(k)
		elif neuron_name == 'EB/([L1,L3],[P,M],[1-4])/s-EB/(L2,[P,M],[1-4])/b-LAL/LVG/b-PB/R4/b':
			sort_list_dict[5] = int(k)
		elif neuron_name == 'EB/([L2,R1],[P,M],[1-4])/s-EB/(L1,[P,M],[1-4])/b-lal/RDG/b-PB/L5/b':
			sort_list_dict[6] = int(k)
		elif neuron_name == 'EB/([R2,L1],[P,M],[1-4])/s-EB/(R1,[P,M],[1-4])/b-LAL/LDG/b-PB/R5/b':
			sort_list_dict[7] = int(k)
		elif neuron_name == 'EB/([R1,R3],[P,M],[1-4])/s-EB/(R2,[P,M],[1-4])/b-lal/RVG/b-PB/L4/b':
			sort_list_dict[8] = int(k)
		elif neuron_name == 'EB/([R4,R2],[P,M],[1-4])/s-EB/(R3,[P,M],[1-4])/b-LAL/LVG/b-PB/R6/b':
			sort_list_dict[9] = int(k)
		elif neuron_name == 'EB/([R3,R5],[P,M],[1-4])/s-EB/(R4,[P,M],[1-4])/b-lal/RDG/b-PB/L3/b':
			sort_list_dict[10] = int(k)
		elif neuron_name == 'EB/([R6,R4],[P,M],[1-4])/s-EB/(R5,[P,M],[1-4])/b-LAL/LDG/b-PB/R7/b':
			sort_list_dict[11] = int(k)
		elif neuron_name == 'EB/([R5,R7],[P,M],[1-4])/s-EB/(R6,[P,M],[1-4])/b-lal/RVG/b-PB/L2/b':
			sort_list_dict[12] = int(k)
		elif neuron_name == 'EB/([R8,R6],[P,M],[1-4])/s-EB/(R7,[P,M],[1-4])/b-LAL/LVG/b-PB/R8/b':
			sort_list_dict[13] = int(k)
		elif neuron_name == 'EB/([R7,L8],[P,M],[1-4])/s-EB/(R8,[P,M],[1-4])/b-lal/RDG/b-PB/L1|R1/b':
			sort_list_dict[14] = int(k)
		elif neuron_name == 'EB/([L7,R8],[P,M],[1-4])/s-EB/(L8,[P,M],[1-4])/b-LAL/LDG/b-PB/R1|L1/b':
			sort_list_dict[15] = int(k)

id_positions = []

for i in range(16):
	sort_list_list.append(sort_list_dict[i])    
	position = list(uids).index(sort_list_dict[i])
	id_positions.append(position)

sort_list[name] = sort_list_list
data_out = data_T[id_positions, :].T

with h5py.File('%s_output_pick.h5' % name, 'w') as f:
	f.create_dataset('/array', data = data_out)


################################### PEI and PEN ###########################################
name = 'PB'
filename = '%s_output.h5' %name
f = h5py.File(filename, 'r')
uids = f['spike_state']['uids'][:]
data_T = f['spike_state']['data'][:].T

sort_list_dict_PEI = {}
sort_list_list_PEI = []

sort_list_dict_PEN = {}
sort_list_list_PEN = []

r = graph.client.query(("select * from "
      "(select name, out()[@class='ArborizationData'][neuropil='{lpu_name}'].regions "
      "as regions from Neuron where name in "
      "(select expand(out()[@class='CircuitModel'].out()"
      "[@class='LeakyIAF'][extern=true].name) from LPU "
      "where name = '{lpu_name}' limit -1)"
      "limit -1) where regions is not null").format(lpu_name=name))
df = conv.pd.as_pandas(r)[0]

for k, n in lpu_name_to_g_nk[name].node.items():
	if n['class'] == 'LeakyIAF' and lpu_name_to_family_map['PB'][n['name']] == 'PB-EB-LAL':
		neuron_name = n['name']
		if neuron_name == 'PB/L8/s-EB/6/b-lal/RVG/b':
			sort_list_dict_PEI[0] = int(k)
		elif neuron_name == 'PB/L7/s-EB/7/b-lal/RDG/b':
			sort_list_dict_PEI[1] = int(k)
		elif neuron_name == 'PB/L6/s-EB/8/b-lal/RVG/b':
			sort_list_dict_PEI[2] = int(k)
		elif neuron_name == 'PB/L5/s-EB/1/b-lal/RDG/b':
			sort_list_dict_PEI[3] = int(k)
		elif neuron_name == 'PB/L4/s-EB/2/b-lal/RVG/b':
			sort_list_dict_PEI[4] = int(k)
		elif neuron_name == 'PB/L3/s-EB/3/b-lal/RDG/b':
			sort_list_dict_PEI[5] = int(k)
		elif neuron_name == 'PB/L2/s-EB/4/b-lal/RVG/b':
			sort_list_dict_PEI[6] = int(k)
		elif neuron_name == 'PB/L1/s-EB/5/b-lal/RDG/b':
			sort_list_dict_PEI[7] = int(k)
		elif neuron_name == 'PB/R1/s-EB/5/b-LAL/LDG/b':
			sort_list_dict_PEI[8] = int(k)
		elif neuron_name == 'PB/R2/s-EB/4/b-LAL/LVG/b':
			sort_list_dict_PEI[9] = int(k)
		elif neuron_name == 'PB/R3/s-EB/3/b-LAL/LDG/b':
			sort_list_dict_PEI[10] = int(k)
		elif neuron_name == 'PB/R4/s-EB/2/b-LAL/LVG/b':
			sort_list_dict_PEI[11] = int(k)
		elif neuron_name == 'PB/R5/s-EB/1/b-LAL/LDG/b':
			sort_list_dict_PEI[12] = int(k)
		elif neuron_name == 'PB/R6/s-EB/8/b-LAL/LVG/b':
			sort_list_dict_PEI[13] = int(k)
		elif neuron_name == 'PB/R7/s-EB/7/b-LAL/LDG/b':
			sort_list_dict_PEI[14] = int(k)
		elif neuron_name == 'PB/R8/s-EB/6/b-LAL/LVG/b':
			sort_list_dict_PEI[15] = int(k)
	elif n['class'] == 'LeakyIAF' and lpu_name_to_family_map['PB'][n['name']] == 'PB-EB-NO':
		neuron_name = n['name']
		if neuron_name == 'PB/L9/s-EB/6/b-no/(1,R)/b':
			sort_list_dict_PEN[0] = int(k)
		elif neuron_name == 'PB/L8/s-EB/7/b-no/(1,R)/b':
			sort_list_dict_PEN[1] = int(k)
		elif neuron_name == 'PB/L7/s-EB/8/b-no/(1,R)/b':
			sort_list_dict_PEN[2] = int(k)
		elif neuron_name == 'PB/L6/s-EB/1/b-no/(1,R)/b':
			sort_list_dict_PEN[3] = int(k)
		elif neuron_name == 'PB/L5/s-EB/2/b-no/(1,R)/b':
			sort_list_dict_PEN[4] = int(k)
		elif neuron_name == 'PB/L4/s-EB/3/b-no/(1,R)/b':
			sort_list_dict_PEN[5] = int(k)
		elif neuron_name == 'PB/L3/s-EB/4/b-no/(1,R)/b':
			sort_list_dict_PEN[6] = int(k)
		elif neuron_name == 'PB/L2/s-EB/5/b-no/(1,R)/b':
			sort_list_dict_PEN[7] = int(k)
		elif neuron_name == 'PB/R2/s-EB/5/b-NO/(1,L)/b':
			sort_list_dict_PEN[8] = int(k)
		elif neuron_name == 'PB/R3/s-EB/6/b-NO/(1,L)/b':
			sort_list_dict_PEN[9] = int(k)
		elif neuron_name == 'PB/R4/s-EB/7/b-NO/(1,L)/b':
			sort_list_dict_PEN[10] = int(k)
		elif neuron_name == 'PB/R5/s-EB/8/b-NO/(1,L)/b':
			sort_list_dict_PEN[11] = int(k)
		elif neuron_name == 'PB/R6/s-EB/1/b-NO/(1,L)/b':
			sort_list_dict_PEN[12] = int(k)
		elif neuron_name == 'PB/R7/s-EB/2/b-NO/(1,L)/b':
			sort_list_dict_PEN[13] = int(k)
		elif neuron_name == 'PB/R8/s-EB/3/b-NO/(1,L)/b':
			sort_list_dict_PEN[14] = int(k)
		elif neuron_name == 'PB/R9/s-EB/4/b-NO/(1,L)/b':
			sort_list_dict_PEN[15] = int(k)


id_positions_PEI = []

for i in range(16):
	sort_list_list_PEI.append(sort_list_dict_PEI[i])    
	position_PEI = list(uids).index(sort_list_dict_PEI[i])
	id_positions_PEI.append(position_PEI)

sort_list['PEI'] = sort_list_list_PEI
data_out = data_T[id_positions_PEI, :].T

with h5py.File('%s_output_pick.h5' % 'PEI', 'w') as f:
	f.create_dataset('/array', data = data_out)



id_positions_PEN = []

for i in range(16):
	sort_list_list_PEN.append(sort_list_dict_PEN[i])    
	position_PEN = list(uids).index(sort_list_dict_PEN[i])
	id_positions_PEN.append(position_PEN)

sort_list['PEN'] = sort_list_list_PEN
data_out = data_T[id_positions_PEN, :].T

with h5py.File('%s_output_pick.h5' % 'PEN', 'w') as f:
	f.create_dataset('/array', data = data_out)




logger = setup_logger(screen=True)
dt = 1e-4
update_interval = None
fontsize = 16
xlim = [0, 1.0]

v_BU = vis.visualizer()

v_BU.add_LPU('BU_output.h5', LPU='BU')
v_BU.add_plot({'type': 'raster', 'uids': [sort_list['BU']],'variable': 'spike_state',
               'yticks': range(1, 1+len(sort_list['BU'])), #'yticklabels': range(len(bu_neurons))
               'yticklabels': []},
              'BU')
#[list(np.arange(753,785, dtype=np.int32))]
v_BU.add_LPU('bu_output.h5', LPU='bu')
v_BU.add_plot({'type': 'raster', 'uids': [sort_list['bu']],'variable': 'spike_state',
             'yticks': range(1, 1+len(sort_list['bu'])), #'yticklabels': range(len(bu_neurons))
             'yticklabels': []},
             'bu')

v_BU.update_interval = update_interval

v_BU.fontsize = fontsize
v_BU.dt = dt
v_BU.xlim = xlim

print ('generating BU image ...')
v_BU.run('BU_output_sort_thousand%s.png' % args.d)


v_EB = vis.visualizer()
v_EB.add_LPU('EB_output.h5', LPU='EB')
v_EB.add_plot({'type': 'raster', 'uids': [sort_list['EB']],'variable': 'spike_state',
               'yticks': range(1, 1+len(sort_list['EB'])), #'yticklabels': range(len(bu_neurons))
               'yticklabels': []},
              'EB')


v_EB.update_interval = update_interval
v_EB.fontsize = fontsize
v_EB.dt = dt
v_EB.xlim = xlim

print ('generating EB image ...')
v_EB.run('EB_output_sort_thousand%s.png' % args.d)




v_PB = vis.visualizer()
v_PB.add_LPU('PB_output.h5', LPU='PEI')
v_PB.add_plot({'type': 'raster', 'uids': [sort_list['PEI']],'variable': 'spike_state',
               'yticks': range(1, 1+len(sort_list['PEI'])), #'yticklabels': range(len(bu_neurons))
               'yticklabels': []},
              'PEI')
v_PB.add_LPU('PB_output.h5', LPU='PEN')
v_PB.add_plot({'type': 'raster', 'uids': [sort_list['PEN']],'variable': 'spike_state',
             'yticks': range(1, 1+len(sort_list['PEN'])), #'yticklabels': range(len(bu_neurons))
             'yticklabels': []},
             'PEN')

v_PB.update_interval = update_interval
v_PB.fontsize = fontsize
v_PB.dt = dt
v_PB.xlim = xlim

print ('generating PB image ...')
v_PB.run('PB_output_sort_thousand%s.png' % args.d)
