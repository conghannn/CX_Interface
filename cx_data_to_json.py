#!/usr/bin/env python

"""
Extract arborization data from CSV files, parse it, and store in JSON files
for loading with OrientDB ETL.
"""

import csv
import json
import re
import sys

import path
from mako.template import Template

from parse_arborization import NeuronArborizationParser

from cx_config import cx_db

# Create cfg JSON files from templates:

for file_in in ['neurons_cfg.json.tmpl', 'neuropils_cfg.json.tmpl', 'arbors_cfg.json.tmpl']:
    file_out = re.sub('\.tmpl', '', file_in)
    with open(file_in, 'r') as f_in, open(file_out, 'w') as f_out:
        in_str = f_in.read()
        out_str = Template(in_str).render(db='remote:localhost'+cx_db)
        f_out.write(out_str)

# File names grouped by neuropil in which neurons' presynaptic terminals
# arborize:
real_data = path.Path('real_neuron_data')
hypo_data = path.Path('hypo_neuron_data')

neuropil_to_file_list = {'BU': hypo_data.files('bu_eb_1.csv'),
                         'bu': hypo_data.files('bu_eb_2.csv'),
                         'FB': real_data.files('fb_local.csv')+\
                         hypo_data.files('fb_local_*.csv'),
                         'EB': real_data.files('eb_lal_pb.csv'),
                         'PB': real_data.files('pb*.csv')+\
                         real_data.files('wed_ps_pb.csv')+\
                         real_data.files('ib_lal_ps_pb.csv')}

# File names grouped by neuron family:
family_to_file_list = {'BU-EB': hypo_data.files('bu_eb_*.csv'),
                       'FB': real_data.files('fb_local.csv')+\
                       hypo_data.files('fb_local_*.csv'),
                       'EB-LAL-PB': real_data.files('eb_lal_pb.csv'),
                       'IB-LAL-PS-PB': real_data.files('ib_lal_ps_pb.csv'),
                       'PB-EB-LAL': real_data.files('pb_eb_lal.csv'),
                       'PB-EB-NO': real_data.files('pb_eb_no.csv'),
                       'PB-FB-CRE': real_data.files('pb_fb_cre.csv'),
                       'PB-FB-LAL': real_data.files('pb_fb_lal*.csv'),
                       'PB-FB-NO': real_data.files('pb_fb_no*.csv'),
                       'PB': real_data.files('pb_local.csv'),
                       'WED-PS-PB': real_data.files('wed_ps_pb.csv')}

# Map file names to neuron family:
file_to_family = {}
for family in family_to_file_list:
    for file_name in family_to_file_list[family]:

        # Sanity check against typos in file list:
        if file_name in file_to_family:
            raise RuntimeError('duplicate file name')
        file_to_family[file_name] = family

# Parse labels into neuron data (i.e., each neuron associated with its label)
# and arborization data (i.e., each arborization associated with its originating
# label):
parser = NeuronArborizationParser()
neuropil_data = [{'name': neuropil} for neuropil in neuropil_to_file_list]
neuron_data = []
arbor_data = []
for neuropil in neuropil_to_file_list.keys():
    for file_name in neuropil_to_file_list[neuropil]:
        with open(file_name, 'r') as f:
            r = csv.reader(f, delimiter=' ')
            for row in r:
                d = {'name': row[0], 'family': file_to_family[file_name],
                     'neuropil': neuropil}

                # Add 'neuropil' attrib to each neuron data entry to enable ETL to
                # link each Neuron node to the appropriate Neuropil node:
                neuron_data.append(d)
                try:
                    tmp = parser.parse(row[0])
                except Exception as e:
                    print file_name, row[0]
                    raise e

                # Add 'neuron' attrib to each arborization data entry to enable
                # ETL to link each ArborizationDAta node to the appropriate
                # Neuron node:
                for a in tmp:
                    a['neuron'] = row[0]
                    arbor_data.append(a)

# Custom JSON encoder that converts sets to lists:
class SetToListEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, set):
            return list(o)
        else:
            return json.JSONEncoder.default(self, o)

# Dump data into JSON files:
with open('neuropils.json', 'w') as f:
    json.dump(neuropil_data, f, indent=4)
with open('neurons.json', 'w') as f:
    json.dump(neuron_data, f, indent=4, cls=SetToListEncoder)
with open('arbors.json', 'w') as f:
    json.dump(arbor_data, f, indent=4, cls=SetToListEncoder)
