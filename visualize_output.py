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

# Select spiking projection neurons:
lpu_name_to_neurons = {}
'''
for name in lpu_name_list:
    lpu_name_to_neurons[name] = \
        sorted([int(k) for k, n in lpu_name_to_g_nk[name].node.items() if \
                     n['class'] != 'port_in_spk' and \
                     n['spiking']])
'''


##### Pick 80 Neurons and sort them for visualization   ######
sort_list = {}
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



#### former method (by yiyin)  , but the order of neuron's regions is not ordered   #####
      
for name in lpu_name_list:
    lpu_name_to_neurons[name] = \
        sorted([int(k) for k, n in lpu_name_to_g_nk[name].node.items() if n['class'] == 'LeakyIAF'])


'''

# Get maps between neuron names and families:
lpu_name_to_family_map = {}
for name in lpu_name_list:
    r = graph.client.query(("select name, family from Neuron where name in "
                "(select expand(out()[@class='CircuitModel'].out()[@class='LeakyIAF'][extern=true].name)"
                           "from LPU where name = '{name}' limit -1)").format(name=name))
    df = conv.pd.as_pandas(r)[0]
    lpu_name_to_family_map[name] = {r['name']:r['family'] for r in df.to_dict('record')}

'''



'''
v_PB = vis.visualizer()
'''

# conf_input = {}
# conf_input['type'] = 'image'
# conf_input['clim'] = [0, 0.02]
# conf_input['ids'] = [range(208)]
# conf_input['shape'] = [1, 208]

# v.add_LPU('PB_input.h5', LPU='PB')
# v.add_plot(conf_input, 'input_PB')

# v.add_LPU('PB_input.h5', LPU='PB')
# v.add_plot({'type':'waveform', 'ids': [[0]]}, 'input_PB')

# v.add_LPU('PB_output_spike.h5',
#           graph=lpu_name_to_g_nk['PB'], LPU='PB')
# v.add_plot({'type': 'raster', 'ids': {0: lpu_name_to_neurons['PB']},
#             'yticks': range(1, 1+len(lpu_name_to_neurons['PB'])), 'yticklabels': range(len(pb_neurons)),
#             'yticklabels': []},
#             'PB', 'Output')
'''

PB_EB_LAL_ids = [i for i, name in enumerate(lpu_name_to_comp_dict['PB']['LeakyIAF']['name']) \
                 if lpu_name_to_family_map['PB'][name] == 'PB-EB-LAL']
PB_EB_LAL_names = [name for i, name in enumerate(lpu_name_to_comp_dict['PB']['LeakyIAF']['name']) \
                   if lpu_name_to_family_map['PB'][name] == 'PB-EB-LAL']
v_PB.add_LPU('PB_output_spike.h5', LPU='PB-EB-LAL')
v_PB.add_plot({'type': 'raster', 'ids': {0: PB_EB_LAL_ids},
               'yticks': range(1, 1+len(PB_EB_LAL_ids)), 
               'yticklabels': []},
              'PB-EB-LAL')

PB_FB_LAL_ids = [i for i, name in enumerate(lpu_name_to_comp_dict['PB']['LeakyIAF']['name']) \
                 if lpu_name_to_family_map['PB'][name] == 'PB-FB-LAL']
v_PB.add_LPU('PB_output_spike.h5', LPU='PB-FB-LAL')
v_PB.add_plot({'type': 'raster', 'ids': {0: PB_FB_LAL_ids},
               'yticks': range(1, 1+len(PB_FB_LAL_ids)),
               'yticklabels': []},
              'PB-FB-LAL')

PB_EB_NO_ids = [i for i, name in enumerate(lpu_name_to_comp_dict['PB']['LeakyIAF']['name']) \
                if lpu_name_to_family_map['PB'][name] == 'PB-EB-NO']
v_PB.add_LPU('PB_output_spike.h5', LPU='PB-EB-NO')
v_PB.add_plot({'type': 'raster', 'ids': {0: PB_EB_NO_ids},
               'yticks': range(1, 1+len(PB_EB_NO_ids)),
               'yticklabels': []},
              'PB-EB-NO')

EB_LAL_PB_ids = [i for i, name in enumerate(lpu_name_to_comp_dict['EB']['LeakyIAF']['name']) \
                if lpu_name_to_family_map['EB'][name] == 'EB-LAL-PB']
v_PB.add_LPU('EB_output_spike.h5', LPU='EB-LAL-PB')
v_PB.add_plot({'type': 'raster', 'ids': {0: EB_LAL_PB_ids},
               'yticks': range(1, 1+len(EB_LAL_PB_ids)),
               'yticklabels': []},
              'EB-LAL-PB')

# PB_local_ids = [i for i, name in enumerate(lpu_name_to_n_dict['PB']['LeakyIAF']['name']) \
#                  if lpu_name_to_family_map['PB'][name] == 'PB']
# v_PB.add_LPU('PB_output_spike.h5',
#           graph=lpu_name_to_g_nk['PB'], LPU='PB Local')
# v_PB.add_plot({'type': 'raster', 'ids': {0: PB_local_ids},
#                'yticks': range(1, 1+len(PB_local_ids)), #'yticklabels': range(len(pb_neurons)),
#                'yticklabels': []},
#               'PB Local', 'Output')

v_PB.update_interval = update_interval
v_PB.fontsize = fontsize
v_PB.dt = dt
v_PB.xlim = xlim
v_PB.run('PB_output_%s.png' % args.d)
'''
logger = setup_logger(screen=True)
dt = 1e-4
update_interval = None
fontsize = 16
xlim = [0, 1.0]

v_BU = vis.visualizer()

v_BU.add_LPU('BU_output.h5', LPU='BU')
v_BU.add_plot({'type': 'raster', 'uids': [sort_list['BU']],'variable': 'spike_state',
               'yticks': range(1, 1+len(lpu_name_to_neurons['BU'])), #'yticklabels': range(len(bu_neurons))
               'yticklabels': []},
              'BU')
#[list(np.arange(753,785, dtype=np.int32))]
v_BU.add_LPU('bu_output.h5', LPU='bu')
v_BU.add_plot({'type': 'raster', 'uids': [sort_list['bu']],'variable': 'spike_state',
             'yticks': range(1, 1+len(lpu_name_to_neurons['bu'])), #'yticklabels': range(len(bu_neurons))
             'yticklabels': []},
             'bu')

v_BU.update_interval = update_interval

v_BU.fontsize = fontsize
v_BU.dt = dt
v_BU.xlim = xlim

print ('generating image ...')
v_BU.run('BU_output_sort_thousand%s.png' % args.d)

# v.add_LPU('FB_output_spike.h5',
#           graph=lpu_name_to_g_nk['FB'], LPU='FB')
# v.add_plot({'type': 'raster', 'ids': {0: lpu_name_to_neurons['FB']},
#             'yticks': range(1, 1+len(lpu_name_to_neurons['FB'])), #'yticklabels': range(len(bu_neurons))
#             'yticklabels': []},
#             'FB', 'Output')
