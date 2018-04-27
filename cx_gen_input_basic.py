#!/usr/bin/env python

"""
Create input signal.
"""

import logging
import sys

import h5py
import numpy as np
from pyorient.ogm import Graph, Config

import neuroarch.models as models

from cx_config import cx_db
graph = Graph(Config.from_url(cx_db, 'admin', 'admin',
                              initial_drop=False))
graph.include(models.Node.registry)
graph.include(models.Relationship.registry)

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('cx')

def create_input(file_name, N, dt=1e-4, dur=1.0, start=0.3, stop=0.6, I_max=0.6):
    Nt = int(dur/dt)
    t  = np.arange(0, dt*Nt, dt)

    I  = np.zeros((Nt, N), dtype=np.float64)
    I[np.logical_and(t>start, t<stop)] = I_max

    with h5py.File(file_name, 'w') as f:
        f.create_dataset('array', (Nt, N),
                         dtype=np.float64,
                         data=I)

# Get the number of LeakyIAF nodes that accept external input:
lpu_name_to_node = {}
lpu_name_to_q = {}
lpu_name_to_df = {}
lpu_name_to_N = {}

for lpu_name in ['BU', 'bu', 'PB']:
    lpu_name_to_node[lpu_name] = \
        graph.LPUs.query(name=lpu_name).one()
    lpu_name_to_q[lpu_name] = \
        lpu_name_to_node[lpu_name].owns(2, ['LeakyIAF'])
    lpu_name_to_df[lpu_name], _ = lpu_name_to_q[lpu_name].get_as('df')
    lpu_name_to_N[lpu_name] = \
        len(lpu_name_to_df[lpu_name][lpu_name_to_df[lpu_name]['extern']])
    logger.info('%s LeakyIAF nodes accepting input in %s found' % \
                (lpu_name_to_N[lpu_name], lpu_name))

dt = 1e-4
dur = 1.0
start = 0.3
stop = 0.6
I_max = 0.6

for lpu_name in ['BU', 'bu', 'PB']:
    logger.info('creating input signal for %s' % lpu_name)
    in_file_name = '%s_input.h5' % lpu_name
    create_input(in_file_name, lpu_name_to_N[lpu_name], 
                 dt, dur, start, stop, I_max)
