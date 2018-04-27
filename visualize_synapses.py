#!/usr/bin/env python

"""
Visualize inferred synapses.
"""

import itertools
import logging
import sys

import numpy as np
import pandas as pd
from pyorient.ogm import Graph, Config

import neuroarch.models as models
import neuroarch.query as query
import neuroarch.conv as conv
import arbor_funcs

from cx_config import cx_db

graph = Graph(Config.from_url(cx_db, 'admin', 'admin',
                              initial_drop=False))
graph.include(models.Node.registry)
graph.include(models.Relationship.registry)

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('cx')


def get_connected_neurons(from_family, to_family):
    r = graph.client.query(("select in()[@class='Neuron'].name as from_name, "
                           "out()[@class='Neuron'].name as to_name "
                           "from (select expand($c) "
                           "let $a = (select * from Neuron where family = '{from_family}'), "
                           "$b = (select * from Neuron where family = '{to_family}'), "
                           "$c = intersect($a.out('SendsTo'),$b.in('SendsTo')) limit -1) "
                           "unwind from_name, to_name").format(from_family=from_family,
                                                               to_family=to_family))
    return conv.pd.as_pandas(r)[0]

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams.update({'figure.figsize': (12, 10)})
# Get neuron families as Pandas DataFrames:
family_list = ['PB', 'PB-EB-LAL', 'PB-EB-NO', 'EB-LAL-PB']
family_to_neurons = {}
# for family in family_list:
#     q = query.QueryWrapper.from_elements(graph, *graph.Neurons.query(family=family).all())
#     family_to_neurons[family] = q.get_as('df')[0]
#     family_to_neurons[family].index = range(len(family_to_neurons[family]))

family_to_neurons['PB'] = \
    pd.DataFrame({'name': [
        'PB/L9/b-PB/L[4-8]/s',
        'PB/L8|R1|R9/b-PB/L[1-9]|R[1-9]/s',
        'PB/L7|R2/b-PB/L[1-9]|R[1-9]/s',
        'PB/L6|R3/b-PB/L[1-9]|R[1-9]/s',
        'PB/L5|R4/b-PB/L[1-9]|R[1-9]/s',
        'PB/R5|L4/b-PB/R[1-9]|L[1-9]/s',
        'PB/R6|L3/b-PB/R[1-9]|L[1-9]/s',
        'PB/R7|L2/b-PB/R[1-9]|L[1-9]/s',
        'PB/R8|L1|L9/b-PB/R[1-9]|L[1-9]/s',
        'PB/R9/b-PB/R[4-8]/s']})

family_to_neurons['PB-EB-LAL'] = \
    pd.DataFrame({'name': [
        'PB/L8/s-EB/6/b-lal/RVG/b',
        'PB/L7/s-EB/7/b-lal/RDG/b',
        'PB/L6/s-EB/8/b-lal/RVG/b',
        'PB/L5/s-EB/1/b-lal/RDG/b',
        'PB/L4/s-EB/2/b-lal/RVG/b',
        'PB/L3/s-EB/3/b-lal/RDG/b',
        'PB/L2/s-EB/4/b-lal/RVG/b',
        'PB/L1/s-EB/5/b-lal/RDG/b',
        'PB/R1/s-EB/5/b-LAL/LDG/b',
        'PB/R2/s-EB/4/b-LAL/LVG/b',
        'PB/R3/s-EB/3/b-LAL/LDG/b',
        'PB/R4/s-EB/2/b-LAL/LVG/b',
        'PB/R5/s-EB/1/b-LAL/LDG/b',
        'PB/R6/s-EB/8/b-LAL/LVG/b',
        'PB/R7/s-EB/7/b-LAL/LDG/b',
        'PB/R8/s-EB/6/b-LAL/LVG/b']})

family_to_neurons['PB-EB-NO'] = \
    pd.DataFrame({'name': [
        'PB/L9/s-EB/6/b-no/(1,R)/b',
        'PB/L8/s-EB/7/b-no/(1,R)/b',
        'PB/L7/s-EB/8/b-no/(1,R)/b',
        'PB/L6/s-EB/1/b-no/(1,R)/b',
        'PB/L5/s-EB/2/b-no/(1,R)/b',
        'PB/L4/s-EB/3/b-no/(1,R)/b',
        'PB/L3/s-EB/4/b-no/(1,R)/b',
        'PB/L2/s-EB/5/b-no/(1,R)/b',
        'PB/R2/s-EB/5/b-NO/(1,L)/b',
        'PB/R3/s-EB/6/b-NO/(1,L)/b',
        'PB/R4/s-EB/7/b-NO/(1,L)/b',
        'PB/R5/s-EB/8/b-NO/(1,L)/b',
        'PB/R6/s-EB/1/b-NO/(1,L)/b',
        'PB/R7/s-EB/2/b-NO/(1,L)/b',
        'PB/R8/s-EB/3/b-NO/(1,L)/b',
        'PB/R9/s-EB/4/b-NO/(1,L)/b']})
                  
family_to_neurons['EB-LAL-PB'] = pd.DataFrame({'name': [
    'EB/(L8,P,[1-4])/s-EB/(L8,P,[1-4])/b-lal/RDG/b-PB/L9/b',
    'EB/([L7,R8],[P,M],[1-4])/s-EB/(L8,[P,M],[1-4])/b-LAL/LDG/b-PB/R1|L1/b',
    'EB/([L8,L6],[P,M],[1-4])/s-EB/(L7,[P,M],[1-4])/b-lal/RVG/b-PB/L8/b',
    'EB/([L5,L7],[P,M],[1-4])/s-EB/(L6,[P,M],[1-4])/b-LAL/LVG/b-PB/R2/b',
    'EB/([L6,L4],[P,M],[1-4])/s-EB/(L5,[P,M],[1-4])/b-lal/RDG/b-PB/L7/b',
    'EB/([L3,L5],[P,M],[1-4])/s-EB/(L4,[P,M],[1-4])/b-LAL/LDG/b-PB/R3/b',
    'EB/([L4,L2],[P,M],[1-4])/s-EB/(L3,[P,M],[1-4])/b-lal/RVG/b-PB/L6/b',
    'EB/([L1,L3],[P,M],[1-4])/s-EB/(L2,[P,M],[1-4])/b-LAL/LVG/b-PB/R4/b',
    'EB/([L2,R1],[P,M],[1-4])/s-EB/(L1,[P,M],[1-4])/b-lal/RDG/b-PB/L5/b',
    'EB/([R2,L1],[P,M],[1-4])/s-EB/(R1,[P,M],[1-4])/b-LAL/LDG/b-PB/R5/b',
    'EB/([R1,R3],[P,M],[1-4])/s-EB/(R2,[P,M],[1-4])/b-lal/RVG/b-PB/L4/b',
    'EB/([R4,R2],[P,M],[1-4])/s-EB/(R3,[P,M],[1-4])/b-LAL/LVG/b-PB/R6/b',
    'EB/([R3,R5],[P,M],[1-4])/s-EB/(R4,[P,M],[1-4])/b-lal/RDG/b-PB/L3/b',
    'EB/([R6,R4],[P,M],[1-4])/s-EB/(R5,[P,M],[1-4])/b-LAL/LDG/b-PB/R7/b',
    'EB/([R5,R7],[P,M],[1-4])/s-EB/(R6,[P,M],[1-4])/b-lal/RVG/b-PB/L2/b',
    'EB/([R8,R6],[P,M],[1-4])/s-EB/(R7,[P,M],[1-4])/b-LAL/LVG/b-PB/R8/b',
    'EB/([R7,L8],[P,M],[1-4])/s-EB/(R8,[P,M],[1-4])/b-lal/RDG/b-PB/L1|R1/b',
    'EB/(R8,P,[1-4])/s-EB/(R8,P,[1-4])/b-LAL/LDG/b-PB/R9/b']})

# Construct adjacency matrices:
conn_neurons_dict = {}
adj_mat_dict = {}
for fam_a, fam_b in itertools.product(family_list, family_list):
    df_a = family_to_neurons[fam_a]
    df_b = family_to_neurons[fam_b]

    mat = np.zeros((len(df_a), len(df_b)), dtype=int)
    conn_neurons = get_connected_neurons(fam_a, fam_b)

    # Don't bother saving matrices with no connection entries:
    if len(conn_neurons) == 0:
        continue
    conn_neurons_dict[(fam_a, fam_b)] = conn_neurons

    for r in conn_neurons.to_dict('records'):
        try:
            i = df_a[df_a['name']==r['from_name']].index[0]
            j = df_b[df_b['name']==r['to_name']].index[0]
        except:
            pass
        else:
            mat[i, j] = 1

    # Reverse colors:
    mat = -1*mat+1

    adj_mat_dict[(fam_a, fam_b)] = mat
                                              
# Visualize inferred connections:
def show_conns(fam_a, fam_b):
    plt.clf()
    plt.imshow(adj_mat_dict[(fam_a, fam_b)],
               interpolation='none', cmap='gray')
    ax = plt.gca()

    neurons_x = family_to_neurons[fam_b].name.tolist()
    neurons_y = family_to_neurons[fam_a].name.tolist()

    # Append consecutive 1-indexed integers to labels:
    xlabels = [str(i+1) for i, _ in enumerate(neurons_x)]
    ylabels = [str(i+1) for i, _ in enumerate(neurons_y)]

    xmin_val, xmax_val, xdiff = 0.0, float(len(xlabels)), 1.0
    ymin_val, ymax_val, ydiff = 0.0, float(len(ylabels)), 1.0

    xlocs = np.arange(xmin_val-xdiff/2, xmax_val-xdiff/2)
    ylocs = np.arange(ymin_val-ydiff/2, ymax_val-ydiff/2)

    ax.xaxis.set_ticks(xlocs, minor=True)
    ax.xaxis.set_ticks(xlocs+xdiff/2)
    #ax.xaxis.set_ticklabels(xlabels, rotation='vertical')
    ax.xaxis.set_ticklabels(xlabels)

    ax.yaxis.set_ticks(ylocs, minor=True)
    ax.yaxis.set_ticks(ylocs+ydiff/2)
    ax.yaxis.set_ticklabels(ylabels)

    ax.set_xlim(xmin_val-xdiff/2, xmax_val-xdiff/2)
    ax.set_ylim(ymin_val-ydiff/2, ymax_val-ydiff/2)

    plt.grid(True, which='minor')
    plt.tight_layout()

for fam_a, fam_b in adj_mat_dict.keys():
    show_conns(fam_a, fam_b)
    plt.savefig(('%s_to_%s_mat.png' % (fam_a, fam_b)).replace('-', '_').lower(),
                bbox_inches='tight', dpi=100)
