#!/usr/bin/env python

"""
Create CX database.
"""

import argparse
import itertools
import logging
import sys

from pyorient.ogm import Graph, Config

import neuroarch.models as models

parser = argparse.ArgumentParser()
parser.add_argument('-d', help='Drop database before creating', action='store_true')
parser.add_argument('-c', help='Clear database', action='store_true')
args = parser.parse_args()
initial_drop = args.d

from cx_config import cx_db
graph = Graph(Config.from_url(cx_db, 'admin', 'admin',
                              initial_drop=initial_drop))
models.create_efficiently(graph, models.Node.registry)
models.create_efficiently(graph, models.Relationship.registry)

if args.c:
    graph.client.command('delete vertex V')
