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
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams.update({'figure.figsize': (12, 12)})

# Get neuron families as Pandas DataFrames:
# PB = 1, PB-EB-LAL = 2, EB-LAL-PB = 3, PB-EB-NO = 4, FB = 5, PB-FB-CRE = 6, 
# PB-FB-LAL = 7, PB-FB-NO = 8, BU-EB = 9

# 1, 2, 3, 4
#family_list = ['PB', 'PB-EB-LAL', 'EB-LAL-PB', 'PB-EB-NO']

# 1, 5, 2, 6, 7, 8, 3
#family_list = ['PB', 'FB', 'PB-EB-LAL', 'PB-FB-CRE', 'PB-FB-LAL', 'PB-FB-NO', 'EB-LAL-PB']

# 9, 3
family_list = ['BU-EB', 'EB-LAL-PB']

family_to_neurons = {}

family_to_neurons['BU-EB'] = \
    pd.DataFrame({'name': [
'BU/L1/s-EB/(LR[1-8],[M,A],1)/b',
'BU/L2/s-EB/(LR[1-8],[M,A],1)/b',
'BU/L3/s-EB/(LR[1-8],[M,A],1)/b',
'BU/L4/s-EB/(LR[1-8],[M,A],1)/b',
'BU/L5/s-EB/(LR[1-8],[M,A],1)/b',
'BU/L6/s-EB/(LR[1-8],[M,A],1)/b',
'BU/L7/s-EB/(LR[1-8],[M,A],1)/b',
'BU/L8/s-EB/(LR[1-8],[M,A],1)/b',
'BU/L9/s-EB/(LR[1-8],[M,A],1)/b',
'BU/L10/s-EB/(LR[1-8],[M,A],1)/b',
'BU/L11/s-EB/(LR[1-8],[M,A],1)/b',
'BU/L12/s-EB/(LR[1-8],[M,A],1)/b',
'BU/L13/s-EB/(LR[1-8],[M,A],1)/b',
'BU/L14/s-EB/(LR[1-8],[M,A],1)/b',
'BU/L15/s-EB/(LR[1-8],[M,A],1)/b',
'BU/L16/s-EB/(LR[1-8],[M,A],1)/b',
'BU/L33/s-EB/(LR[1-8],A,[3,4])/b',
'BU/L34/s-EB/(LR[1-8],A,[3,4])/b',
'BU/L35/s-EB/(LR[1-8],A,[3,4])/b',
'BU/L36/s-EB/(LR[1-8],A,[3,4])/b',
'BU/L37/s-EB/(LR[1-8],A,[3,4])/b',
'BU/L38/s-EB/(LR[1-8],A,[3,4])/b',
'BU/L39/s-EB/(LR[1-8],A,[3,4])/b',
'BU/L40/s-EB/(LR[1-8],A,[3,4])/b',
'BU/L41/s-EB/(LR[1-8],A,[3,4])/b',
'BU/L42/s-EB/(LR[1-8],A,[3,4])/b',
'BU/L43/s-EB/(LR[1-8],A,[3,4])/b',
'BU/L44/s-EB/(LR[1-8],A,[3,4])/b',
'BU/L45/s-EB/(LR[1-8],A,[3,4])/b',
'BU/L46/s-EB/(LR[1-8],A,[3,4])/b',
'BU/L47/s-EB/(LR[1-8],A,[3,4])/b',
'BU/L48/s-EB/(LR[1-8],A,[3,4])/b',
'BU/L17/s-EB/(LR[1-8],A,[1,2])/b',
'BU/L18/s-EB/(LR[1-8],A,[1,2])/b',
'BU/L19/s-EB/(LR[1-8],A,[1,2])/b',
'BU/L20/s-EB/(LR[1-8],A,[1,2])/b',
'BU/L21/s-EB/(LR[1-8],A,[1,2])/b',
'BU/L22/s-EB/(LR[1-8],A,[1,2])/b',
'BU/L23/s-EB/(LR[1-8],A,[1,2])/b',
'BU/L24/s-EB/(LR[1-8],A,[1,2])/b',
'BU/L25/s-EB/(LR[1-8],A,[1,2])/b',
'BU/L26/s-EB/(LR[1-8],A,[1,2])/b',
'BU/L27/s-EB/(LR[1-8],A,[1,2])/b',
'BU/L28/s-EB/(LR[1-8],A,[1,2])/b',
'BU/L29/s-EB/(LR[1-8],A,[1,2])/b',
'BU/L30/s-EB/(LR[1-8],A,[1,2])/b',
'BU/L31/s-EB/(LR[1-8],A,[1,2])/b',
'BU/L32/s-EB/(LR[1-8],A,[1,2])/b',
'BU/L49/s-EB/(LR[1-8],A,4)/b',
'BU/L50/s-EB/(LR[1-8],A,4)/b',
'BU/L51/s-EB/(LR[1-8],A,4)/b',
'BU/L52/s-EB/(LR[1-8],A,4)/b',
'BU/L53/s-EB/(LR[1-8],A,4)/b',
'BU/L54/s-EB/(LR[1-8],A,4)/b',
'BU/L55/s-EB/(LR[1-8],A,4)/b',
'BU/L56/s-EB/(LR[1-8],A,4)/b',
'BU/L57/s-EB/(LR[1-8],A,4)/b',
'BU/L58/s-EB/(LR[1-8],A,4)/b',
'BU/L59/s-EB/(LR[1-8],A,4)/b',
'BU/L60/s-EB/(LR[1-8],A,4)/b',
'BU/L61/s-EB/(LR[1-8],A,4)/b',
'BU/L62/s-EB/(LR[1-8],A,4)/b',
'BU/L63/s-EB/(LR[1-8],A,4)/b',
'BU/L64/s-EB/(LR[1-8],A,4)/b',
'BU/L65/s-EB/(LR[1-8],M,4)/b',
'BU/L66/s-EB/(LR[1-8],M,4)/b',
'BU/L67/s-EB/(LR[1-8],M,4)/b',
'BU/L68/s-EB/(LR[1-8],M,4)/b',
'BU/L69/s-EB/(LR[1-8],M,4)/b',
'BU/L70/s-EB/(LR[1-8],M,4)/b',
'BU/L71/s-EB/(LR[1-8],M,4)/b',
'BU/L72/s-EB/(LR[1-8],M,4)/b',
'BU/L73/s-EB/(LR[1-8],M,4)/b',
'BU/L74/s-EB/(LR[1-8],M,4)/b',
'BU/L75/s-EB/(LR[1-8],M,4)/b',
'BU/L76/s-EB/(LR[1-8],M,4)/b',
'BU/L77/s-EB/(LR[1-8],M,4)/b',
'BU/L78/s-EB/(LR[1-8],M,4)/b',
'BU/L79/s-EB/(LR[1-8],M,4)/b',
'BU/L80/s-EB/(LR[1-8],M,4)/b',
'bu/R1/s-EB/(LR[1-8],[M,A],1)/b',
'bu/R2/s-EB/(LR[1-8],[M,A],1)/b',
'bu/R3/s-EB/(LR[1-8],[M,A],1)/b',
'bu/R4/s-EB/(LR[1-8],[M,A],1)/b',
'bu/R5/s-EB/(LR[1-8],[M,A],1)/b',
'bu/R6/s-EB/(LR[1-8],[M,A],1)/b',
'bu/R7/s-EB/(LR[1-8],[M,A],1)/b',
'bu/R8/s-EB/(LR[1-8],[M,A],1)/b',
'bu/R9/s-EB/(LR[1-8],[M,A],1)/b',
'bu/R10/s-EB/(LR[1-8],[M,A],1)/b',
'bu/R11/s-EB/(LR[1-8],[M,A],1)/b',
'bu/R12/s-EB/(LR[1-8],[M,A],1)/b',
'bu/R13/s-EB/(LR[1-8],[M,A],1)/b',
'bu/R14/s-EB/(LR[1-8],[M,A],1)/b',
'bu/R15/s-EB/(LR[1-8],[M,A],1)/b',
'bu/R16/s-EB/(LR[1-8],[M,A],1)/b',
'bu/R33/s-EB/(LR[1-8],A,[3,4])/b',
'bu/R34/s-EB/(LR[1-8],A,[3,4])/b',
'bu/R35/s-EB/(LR[1-8],A,[3,4])/b',
'bu/R36/s-EB/(LR[1-8],A,[3,4])/b',
'bu/R37/s-EB/(LR[1-8],A,[3,4])/b',
'bu/R38/s-EB/(LR[1-8],A,[3,4])/b',
'bu/R39/s-EB/(LR[1-8],A,[3,4])/b',
'bu/R40/s-EB/(LR[1-8],A,[3,4])/b',
'bu/R41/s-EB/(LR[1-8],A,[3,4])/b',
'bu/R42/s-EB/(LR[1-8],A,[3,4])/b',
'bu/R43/s-EB/(LR[1-8],A,[3,4])/b',
'bu/R44/s-EB/(LR[1-8],A,[3,4])/b',
'bu/R45/s-EB/(LR[1-8],A,[3,4])/b',
'bu/R46/s-EB/(LR[1-8],A,[3,4])/b',
'bu/R47/s-EB/(LR[1-8],A,[3,4])/b',
'bu/R48/s-EB/(LR[1-8],A,[3,4])/b',
'bu/R17/s-EB/(LR[1-8],A,[1,2])/b',
'bu/R18/s-EB/(LR[1-8],A,[1,2])/b',
'bu/R19/s-EB/(LR[1-8],A,[1,2])/b',
'bu/R20/s-EB/(LR[1-8],A,[1,2])/b',
'bu/R21/s-EB/(LR[1-8],A,[1,2])/b',
'bu/R22/s-EB/(LR[1-8],A,[1,2])/b',
'bu/R23/s-EB/(LR[1-8],A,[1,2])/b',
'bu/R24/s-EB/(LR[1-8],A,[1,2])/b',
'bu/R25/s-EB/(LR[1-8],A,[1,2])/b',
'bu/R26/s-EB/(LR[1-8],A,[1,2])/b',
'bu/R27/s-EB/(LR[1-8],A,[1,2])/b',
'bu/R28/s-EB/(LR[1-8],A,[1,2])/b',
'bu/R29/s-EB/(LR[1-8],A,[1,2])/b',
'bu/R30/s-EB/(LR[1-8],A,[1,2])/b',
'bu/R31/s-EB/(LR[1-8],A,[1,2])/b',
'bu/R32/s-EB/(LR[1-8],A,[1,2])/b',
'bu/R49/s-EB/(LR[1-8],A,4)/b',
'bu/R50/s-EB/(LR[1-8],A,4)/b',
'bu/R51/s-EB/(LR[1-8],A,4)/b',
'bu/R52/s-EB/(LR[1-8],A,4)/b',
'bu/R53/s-EB/(LR[1-8],A,4)/b',
'bu/R54/s-EB/(LR[1-8],A,4)/b',
'bu/R55/s-EB/(LR[1-8],A,4)/b',
'bu/R56/s-EB/(LR[1-8],A,4)/b',
'bu/R57/s-EB/(LR[1-8],A,4)/b',
'bu/R58/s-EB/(LR[1-8],A,4)/b',
'bu/R59/s-EB/(LR[1-8],A,4)/b',
'bu/R60/s-EB/(LR[1-8],A,4)/b',
'bu/R61/s-EB/(LR[1-8],A,4)/b',
'bu/R62/s-EB/(LR[1-8],A,4)/b',
'bu/R63/s-EB/(LR[1-8],A,4)/b',
'bu/R64/s-EB/(LR[1-8],A,4)/b',
'bu/R65/s-EB/(LR[1-8],M,4)/b',
'bu/R66/s-EB/(LR[1-8],M,4)/b',
'bu/R67/s-EB/(LR[1-8],M,4)/b',
'bu/R68/s-EB/(LR[1-8],M,4)/b',
'bu/R69/s-EB/(LR[1-8],M,4)/b',
'bu/R70/s-EB/(LR[1-8],M,4)/b',
'bu/R71/s-EB/(LR[1-8],M,4)/b',
'bu/R72/s-EB/(LR[1-8],M,4)/b',
'bu/R73/s-EB/(LR[1-8],M,4)/b',
'bu/R74/s-EB/(LR[1-8],M,4)/b',
'bu/R75/s-EB/(LR[1-8],M,4)/b',
'bu/R76/s-EB/(LR[1-8],M,4)/b',
'bu/R77/s-EB/(LR[1-8],M,4)/b',
'bu/R78/s-EB/(LR[1-8],M,4)/b',
'bu/R79/s-EB/(LR[1-8],M,4)/b',
'bu/R80/s-EB/(LR[1-8],M,4)/b']})

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

family_to_neurons['FB'] = \
    pd.DataFrame({'name': [
        'FB/(3,L4)/s-FB/(3,R1)/b',
        'FB/(3,L3)/s-FB/(3,R2)/b',
        'FB/(3,L2)/s-FB/(3,R3)/b',
        'FB/(3,L1)/s-FB/(3,R4)/b',
        'FB/(3,R4)/s-FB/(3,L1)/b',
        'FB/(3,R3)/s-FB/(3,L2)/b',
        'FB/(3,R2)/s-FB/(3,L3)/b',
        'FB/(3,R1)/s-FB/(3,L4)/b',
'FB/(1,L4)/s-FB/(1,R1)/b',
'FB/(1,L3)/s-FB/(1,R2)/b',
'FB/(1,L2)/s-FB/(1,R3)/b',
'FB/(1,L1)/s-FB/(1,R4)/b',
'FB/(1,R4)/s-FB/(1,L1)/b',
'FB/(1,R3)/s-FB/(1,L2)/b',
'FB/(1,R2)/s-FB/(1,L3)/b',
'FB/(1,R1)/s-FB/(1,L4)/b',
'FB/(2,L4)/s-FB/(2,R1)/b',
'FB/(2,L3)/s-FB/(2,R2)/b',
'FB/(2,L2)/s-FB/(2,R3)/b',
'FB/(2,L1)/s-FB/(2,R4)/b',
'FB/(2,R4)/s-FB/(2,L1)/b',
'FB/(2,R3)/s-FB/(2,L2)/b',
'FB/(2,R2)/s-FB/(2,L3)/b',
'FB/(2,R1)/s-FB/(2,L4)/b',
'FB/(4,L4)/s-FB/(4,R1)/b',
'FB/(4,L3)/s-FB/(4,R2)/b',
'FB/(4,L2)/s-FB/(4,R3)/b',
'FB/(4,L1)/s-FB/(4,R4)/b',
'FB/(4,R4)/s-FB/(4,L1)/b',
'FB/(4,R3)/s-FB/(4,L2)/b',
'FB/(4,R2)/s-FB/(4,L3)/b',
'FB/(4,R1)/s-FB/(4,L4)/b',
'FB/(5,L4)/s-FB/(5,R1)/b',
'FB/(5,L3)/s-FB/(5,R2)/b',
'FB/(5,L2)/s-FB/(5,R3)/b',
'FB/(5,L1)/s-FB/(5,R4)/b',
'FB/(5,R4)/s-FB/(5,L1)/b',
'FB/(5,R3)/s-FB/(5,L2)/b',
'FB/(5,R2)/s-FB/(5,L3)/b',
'FB/(5,R1)/s-FB/(5,L4)/b',
'FB/(1,L4)/s-FB/(1,L3)/b',
'FB/(1,L3)/s-FB/(1,L2)/b',
'FB/(1,L2)/s-FB/(1,L1)/b',
'FB/(1,L1)/s-FB/(1,R1)/b',
'FB/(1,R1)/s-FB/(1,R2)/b',
'FB/(1,R2)/s-FB/(1,R3)/b',
'FB/(1,R3)/s-FB/(1,R4)/b',
'FB/(2,L4)/s-FB/(2,L3)/b',
'FB/(2,L3)/s-FB/(2,L2)/b',
'FB/(2,L2)/s-FB/(2,L1)/b',
'FB/(2,L1)/s-FB/(2,R1)/b',
'FB/(2,R1)/s-FB/(2,R2)/b',
'FB/(2,R2)/s-FB/(2,R3)/b',
'FB/(2,R3)/s-FB/(2,R4)/b',
'FB/(3,L4)/s-FB/(3,L3)/b',
'FB/(3,L3)/s-FB/(3,L2)/b',
'FB/(3,L2)/s-FB/(3,L1)/b',
'FB/(3,L1)/s-FB/(3,R1)/b',
'FB/(3,R1)/s-FB/(3,R2)/b',
'FB/(3,R2)/s-FB/(3,R3)/b',
'FB/(3,R3)/s-FB/(3,R4)/b',
'FB/(4,L4)/s-FB/(4,L3)/b',
'FB/(4,L3)/s-FB/(4,L2)/b',
'FB/(4,L2)/s-FB/(4,L1)/b',
'FB/(4,L1)/s-FB/(4,R1)/b',
'FB/(4,R1)/s-FB/(4,R2)/b',
'FB/(4,R2)/s-FB/(4,R3)/b',
'FB/(4,R3)/s-FB/(4,R4)/b',
'FB/(5,L4)/s-FB/(5,L3)/b',
'FB/(5,L3)/s-FB/(5,L2)/b',
'FB/(5,L2)/s-FB/(5,L1)/b',
'FB/(5,L1)/s-FB/(5,R1)/b',
'FB/(5,R1)/s-FB/(5,R2)/b',
'FB/(5,R2)/s-FB/(5,R3)/b',
'FB/(5,R3)/s-FB/(5,R4)/b',
'FB/(5,L1)/s-FB/(4,L1)/b',
'FB/(4,L1)/s-FB/(3,L1)/b',
'FB/(3,L1)/s-FB/(2,L1)/b',
'FB/(2,L1)/s-FB/(1,L1)/b',
'FB/(5,L2)/s-FB/(4,L2)/b',
'FB/(4,L2)/s-FB/(3,L2)/b',
'FB/(3,L2)/s-FB/(2,L2)/b',
'FB/(2,L2)/s-FB/(1,L2)/b',
'FB/(5,L3)/s-FB/(4,L3)/b',
'FB/(4,L3)/s-FB/(3,L3)/b',
'FB/(3,L3)/s-FB/(2,L3)/b',
'FB/(2,L3)/s-FB/(1,L3)/b',
'FB/(5,L4)/s-FB/(4,L4)/b',
'FB/(4,L4)/s-FB/(3,L4)/b',
'FB/(3,L4)/s-FB/(2,L4)/b',
'FB/(2,L4)/s-FB/(1,L4)/b',
'FB/(5,R1)/s-FB/(4,R1)/b',
'FB/(4,R1)/s-FB/(3,R1)/b',
'FB/(3,R1)/s-FB/(2,R1)/b',
'FB/(2,R1)/s-FB/(1,R1)/b',
'FB/(5,R2)/s-FB/(4,R2)/b',
'FB/(4,R2)/s-FB/(3,R2)/b',
'FB/(3,R2)/s-FB/(2,R2)/b',
'FB/(2,R2)/s-FB/(1,R2)/b',
'FB/(5,R3)/s-FB/(4,R3)/b',
'FB/(4,R3)/s-FB/(3,R3)/b',
'FB/(3,R3)/s-FB/(2,R3)/b',
'FB/(2,R3)/s-FB/(1,R3)/b',
'FB/(5,R4)/s-FB/(4,R4)/b',
'FB/(4,R4)/s-FB/(3,R4)/b',
'FB/(3,R4)/s-FB/(2,R4)/b',
'FB/(2,R4)/s-FB/(1,R4)/b',
'FB/(1,L1)/sb-FB/(8,L1)/sb',
'FB/(1,L2)/sb-FB/(8,L2)/sb',
'FB/(1,L3)/sb-FB/(8,L3)/sb',
'FB/(1,L4)/sb-FB/(8,L4)/sb',
'FB/(1,R1)/sb-FB/(8,R1)/sb',
'FB/(1,R2)/sb-FB/(8,R2)/sb',
'FB/(1,R3)/sb-FB/(8,R3)/sb',
'FB/(1,R4)/sb-FB/(8,R4)/sb',
'FB/(2,L1)/sb-FB/(7,L1)/sb',
'FB/(2,L2)/sb-FB/(7,L2)/sb',
'FB/(2,L3)/sb-FB/(7,L3)/sb',
'FB/(2,L4)/sb-FB/(7,L4)/sb',
'FB/(2,R1)/sb-FB/(7,R1)/sb',
'FB/(2,R2)/sb-FB/(7,R2)/sb',
'FB/(2,R3)/sb-FB/(7,R3)/sb',
'FB/(2,R4)/sb-FB/(7,R4)/sb']})

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

family_to_neurons['PB-FB-CRE'] = pd.DataFrame({'name': [
'PB/L1/s-FB/([3-4],L4)/s-CRE/LRB/b',
'PB/R1/s-FB/([3-4],L4)/s-CRE/LRB/b',
'PB/R2/s-FB/([3-4],L3)/s-CRE/LRB/b',
'PB/R3/s-FB/([3-4],L2)/s-CRE/LRB/b',
'PB/R4/s-FB/([3-4],L1)/s-CRE/LRB/b',
'PB/R4/s-FB/([3-4],R1)/s-CRE/LRB/b',
'PB/R5/s-FB/([3-4],R2)/s-CRE/LRB/b',
'PB/R6/s-FB/([3-4],R3)/s-CRE/LRB/b',
'PB/R7/s-FB/([3-4],R4)/s-CRE/LRB/b',
'PB/L7/s-FB/([3-4],L4)/s-cre/RRB/b',
'PB/L6/s-FB/([3-4],L3)/s-cre/RRB/b',
'PB/L5/s-FB/([3-4],L2)/s-cre/RRB/b',
'PB/L4/s-FB/([3-4],L1)/s-cre/RRB/b',
'PB/L4/s-FB/([3-4],R1)/s-cre/RRB/b',
'PB/L3/s-FB/([3-4],R2)/s-cre/RRB/b',
'PB/L2/s-FB/([3-4],R3)/s-cre/RRB/b',
'PB/L1/s-FB/([3-4],R4)/s-cre/RRB/b',
'PB/R1/s-FB/([3-4],R4)/s-cre/RRB/b']})

family_to_neurons['PB-FB-LAL'] = pd.DataFrame({'name': [
'PB/L1/s-FB/(2,L[3-4])/s-LAL/LHB/b',
'PB/R1|L1/s-FB/(2,L[3-4])/s-LAL/LHB/b',
'PB/R1/s-FB/(2,L[2-3])/s-LAL/LHB/b',
'PB/R2/s-FB/(2,L[1-2])/s-LAL/LHB/b',
'PB/R3/s-FB/(2,[L1,R1])/s-LAL/LHB/b',
'PB/R4/s-FB/(2,R[1-2])/s-LAL/LHB/b',
'PB/R5/s-FB/(2,R[2-3])/s-LAL/LHB/b',
'PB/R6/s-FB/(2,R[3-4])/s-LAL/LHB/b',
'PB/R7/s-FB/(2,R[3-4])/s-LAL/LHB/b',
'PB/L7/s-FB/(2,L[3-4])/s-lal/RHB/b',
'PB/L6/s-FB/(2,L[3-4])/s-lal/RHB/b',
'PB/L5/s-FB/(2,L[2-3])/s-lal/RHB/b',
'PB/L4/s-FB/(2,L[1-2])/s-lal/RHB/b',
'PB/L3/s-FB/(2,[R1,L1])/s-lal/RHB/b',
'PB/L2/s-FB/(2,R[1-2])/s-lal/RHB/b',
'PB/L1/s-FB/(2,R[2-3])/s-lal/RHB/b',
'PB/L1|R1/s-FB/(2,R[3-4])/s-lal/RHB/b',
'PB/R1/s-FB/(2,R[3-4])/s-lal/RHB/b',
'PB/L1/s-FB/([1-4],L[3-4])/s-LAL/LHB/b',
'PB/L1|R1/s-FB/([1-4],L[2-3])/s-LAL/LHB/b',
'PB/R1/s-FB/([1-4],L[1-2])/s-LAL/LHB/b',
'PB/R2/s-FB/([1-4],[L1,R1])/s-LAL/LHB/b',
'PB/R3/s-FB/([1-4],R[1-2])/s-LAL/LHB/b',
'PB/R4/s-FB/([1-4],R[2-3])/s-LAL/LHB/b',
'PB/R5/s-FB/([1-4],R[3-4])/s-LAL/LHB/b',
'PB/R6/s-FB/([1-4],R[3-4])/s-LAL/LHB/b',
'PB/L6/s-FB/([1-4],L[3-4])/s-lal/RHB/b',
'PB/L5/s-FB/([1-4],L[3-4])/s-lal/RHB/b',
'PB/L4/s-FB/([1-4],L[2-3])/s-lal/RHB/b',
'PB/L3/s-FB/([1-4],L[1-2])/s-lal/RHB/b',
'PB/L2/s-FB/([1-4],[L1,R1])/s-lal/RHB/b',
'PB/L1/s-FB/([1-4],R[1-2])/s-lal/RHB/b',
'PB/L1|R1/s-FB/([1-4],R[2-3])/s-lal/RHB/b',
'PB/R1/s-FB/([1-4],R[3-4])/s-lal/RHB/b',
'PB/L3/s-FB/([1-4],L[3-4])/s-LAL/LHB/b-lal/RHB/b',
'PB/L1/s-FB/([1-4],L[1-2])/s-LAL/LHB/b-lal/RHB/b',
'PB/R1/s-FB/([1-4],R[1-2])/s-LAL/LHB/b-lal/RHB/b',
'PB/R3/s-FB/([1-4],R[3-4])/s-LAL/LHB/b-lal/RHB/b']})

family_to_neurons['PB-FB-NO'] = pd.DataFrame({'name': [
'PB/L2/s-FB/(1,R4)/b-no/(3,RP)/b',
'PB/L3/s-FB/(1,R4)/b-no/(3,RP)/b',
'PB/L4/s-FB/(1,R3)/b-no/(3,RP)/b',
'PB/L5/s-FB/(1,R2)/b-no/(3,RP)/b',
'PB/L6/s-FB/(1,R1)|(1,L1)/b-no/(3,RP)/b',
'PB/L7/s-FB/(1,L2)/b-no/(3,RP)/b',
'PB/L8/s-FB/(1,L3)/b-no/(3,RP)/b',
'PB/L9/s-FB/(1,L4)/b-no/(3,RP)/b',
'PB/R2/s-FB/(1,L4)/b-NO/(3,LP)/b',
'PB/R3/s-FB/(1,L4)/b-NO/(3,LP)/b',
'PB/R4/s-FB/(1,L3)/b-NO/(3,LP)/b',
'PB/R5/s-FB/(1,L2)/b-NO/(3,LP)/b',
'PB/R6/s-FB/(1,L1)|(1,R1)/b-NO/(3,LP)/b',
'PB/R7/s-FB/(1,R2)/b-NO/(3,LP)/b',
'PB/R8/s-FB/(1,R3)/b-NO/(3,LP)/b',
'PB/R9/s-FB/(1,R4)/b-NO/(3,LP)/b',
'PB/L2/s-FB/(1,R4)/b-no/(3,RM)/b',
'PB/L3/s-FB/(1,R4)/b-no/(3,RM)/b',
'PB/L4/s-FB/(1,R3)/b-no/(3,RM)/b',
'PB/L5/s-FB/(1,R2)/b-no/(3,RM)/b',
'PB/L6/s-FB/(1,R1)|(1,L1)/b-no/(3,RM)/b',
'PB/L7/s-FB/(1,L2)/b-no/(3,RM)/b',
'PB/L8/s-FB/(1,L3)/b-no/(3,RM)/b',
'PB/L9/s-FB/(1,L4)/b-no/(3,RM)/b',
'PB/R2/s-FB/(1,L4)/b-NO/(3,LM)/b',
'PB/R3/s-FB/(1,L4)/b-NO/(3,LM)/b',
'PB/R4/s-FB/(1,L3)/b-NO/(3,LM)/b',
'PB/R5/s-FB/(1,L2)/b-NO/(3,LM)/b',
'PB/R6/s-FB/(1,L1)|(1,R1)/b-NO/(3,LM)/b',
'PB/R7/s-FB/(1,R2)/b-NO/(3,LM)/b',
'PB/R8/s-FB/(1,R3)/b-NO/(3,LM)/b',
'PB/R9/s-FB/(1,R4)/b-NO/(3,LM)/b',
'PB/L2/s-FB/(2,R4)/b-no/(3,RA)/b',
'PB/L3/s-FB/(2,R4)/b-no/(3,RA)/b',
'PB/L4/s-FB/(2,R3)/b-no/(3,RA)/b',
'PB/L5/s-FB/(2,R2)/b-no/(3,RA)/b',
'PB/L6/s-FB/(2,R1)|(1,L1)/b-no/(3,RA)/b',
'PB/L7/s-FB/(2,L2)/b-no/(3,RA)/b',
'PB/L8/s-FB/(2,L3)/b-no/(3,RA)/b',
'PB/L9/s-FB/(2,L4)/b-no/(3,RA)/b',
'PB/R2/s-FB/(2,L4)/b-NO/(3,LA)/b',
'PB/R3/s-FB/(2,L4)/b-NO/(3,LA)/b',
'PB/R4/s-FB/(2,L3)/b-NO/(3,LA)/b',
'PB/R5/s-FB/(2,L2)/b-NO/(3,LA)/b',
'PB/R6/s-FB/(2,L1)|(1,R1)/b-NO/(3,LA)/b',
'PB/R7/s-FB/(2,R2)/b-NO/(3,LA)/b',
'PB/R8/s-FB/(2,R3)/b-NO/(3,LA)/b',
'PB/R9/s-FB/(2,R4)/b-NO/(3,LA)/b',
'PB/L2/s-FB/(3,R4)/b-no/(2,RD)/b',
'PB/L3/s-FB/(3,R4)/b-no/(2,RD)/b',
'PB/L4/s-FB/(3,R3)/b-no/(2,RD)/b',
'PB/L5/s-FB/(3,R2)/b-no/(2,RD)/b',
'PB/L6/s-FB/(3,R1)|(1,L1)/b-no/(2,RD)/b',
'PB/L7/s-FB/(3,L2)/b-no/(2,RD)/b',
'PB/L8/s-FB/(3,L3)/b-no/(2,RD)/b',
'PB/L9/s-FB/(3,L4)/b-no/(2,RD)/b',
'PB/R2/s-FB/(3,L4)/b-NO/(2,LD)/b',
'PB/R3/s-FB/(3,L4)/b-NO/(2,LD)/b',
'PB/R4/s-FB/(3,L3)/b-NO/(2,LD)/b',
'PB/R5/s-FB/(3,L2)/b-NO/(2,LD)/b',
'PB/R6/s-FB/(3,L1)|(1,R1)/b-NO/(2,LD)/b',
'PB/R7/s-FB/(3,R2)/b-NO/(2,LD)/b',
'PB/R8/s-FB/(3,R3)/b-NO/(2,LD)/b',
'PB/R9/s-FB/(3,R4)/b-NO/(2,LD)/b',
'PB/L2/s-FB/(3,R4)/b-no/(2,RV)/b',
'PB/L3/s-FB/(3,R4)/b-no/(2,RV)/b',
'PB/L4/s-FB/(3,R3)/b-no/(2,RV)/b',
'PB/L5/s-FB/(3,R2)/b-no/(2,RV)/b',
'PB/L6/s-FB/(3,R1)|(1,L1)/b-no/(2,RV)/b',
'PB/L7/s-FB/(3,L2)/b-no/(2,RV)/b',
'PB/L8/s-FB/(3,L3)/b-no/(2,RV)/b',
'PB/L9/s-FB/(3,L4)/b-no/(2,RV)/b',
'PB/R2/s-FB/(3,L4)/b-NO/(2,LV)/b',
'PB/R3/s-FB/(3,L4)/b-NO/(2,LV)/b',
'PB/R4/s-FB/(3,L3)/b-NO/(2,LV)/b',
'PB/R5/s-FB/(3,L2)/b-NO/(2,LV)/b',
'PB/R6/s-FB/(3,L1)|(1,R1)/b-NO/(2,LV)/b',
'PB/R7/s-FB/(3,R2)/b-NO/(2,LV)/b',
'PB/R8/s-FB/(3,R3)/b-NO/(2,LV)/b',
'PB/R9/s-FB/(3,R4)/b-NO/(2,LV)/b']})

# Construct adjacency matrices:
fam_names = family_list
fam_len = [len(family_to_neurons[k]) for k in family_list]

fam_end_inds = np.cumsum(fam_len)
tmp = np.hstack((0, fam_end_inds))
fam_name_to_interval = {name:slice(tmp[i], tmp[i+1]) for name, i in zip(fam_names, range(len(fam_names)))}
conn_neurons_dict = {}
adj_mat_dict = {}
big_mat = np.ones((sum(fam_len), sum(fam_len)), float)
for fam_a, fam_b in itertools.product(family_list, family_list):
    df_a = family_to_neurons[fam_a]
    df_b = family_to_neurons[fam_b]

    rows = len(df_a)
    cols = len(df_b)

    mat = np.zeros((rows, cols), dtype=int)
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
                                              
for k, v in adj_mat_dict.items():
    fam_a, fam_b = k
    big_mat[fam_name_to_interval[fam_a], fam_name_to_interval[fam_b]] = adj_mat_dict[k]
print sorted(fam_name_to_interval.items(), cmp=lambda a,b: cmp(a[1].start, b[1].start))

big_mat[big_mat==0] = 0.25
plt.imshow(big_mat, interpolation='none', cmap='gray', vmin=0, vmax=1)
ax = plt.gca()
plt.autoscale(False)

xmin_val, xmax_val, xdiff = 0.0, float(big_mat.shape[1]), 1.0
ymin_val, ymax_val, ydiff = 0.0, float(big_mat.shape[0]), 1.0

xlocs = np.arange(xmin_val-xdiff/2, xmax_val-xdiff/2)
ylocs = np.arange(ymin_val-ydiff/2, ymax_val-ydiff/2)

ax.xaxis.set_ticks(xlocs, minor=False)
plt.setp(ax.get_xticklabels(), visible=False)

ax.yaxis.set_ticks(ylocs, minor=False)
plt.setp(ax.get_yticklabels(), visible=False)

# Draw grid:
ax.grid(True, which='major')
ax.set_axisbelow(False)

# Draw lines between regions:
for sx in fam_name_to_interval.values():
    for sy in fam_name_to_interval.values():
        plt.plot([sx.stop-xdiff/2, sx.stop-xdiff/2], [sy.start-ydiff/2, sy.stop-ydiff/2], 'k')
        plt.plot([sx.start-xdiff/2, sx.stop-xdiff/2], [sy.stop-ydiff/2, sy.stop-ydiff/2], 'k')
plt.savefig('big_mat.png', bbox_inches='tight', dpi=300)
