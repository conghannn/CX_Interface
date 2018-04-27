#!/usr/bin/env python

"""
Infer connections between neurons from arborizations.
"""

import csv
import itertools

import networkx as nx
import voluptuous as vol

from parse_arborization import NeuronArborizationParser

def is_arbor(a):
    """
    A valid arborization dict must adhere to the following schema:
    {'neurite': set, 'neuropil': str, 'regions': set}
    """

    try:
        is_arbor.v(a)
    except:
        return False
    else:
        return True
is_arbor.v = vol.Schema({vol.Required('neurite'): set,
                         vol.Required('neuropil'): vol.All(str, vol.Length(min=1)),
                         vol.Required('regions'): set})

def flow_direction(arbor0, arbor1):
    """
    Return direction of information flow between arbors.

    Returns
    -------
    result : set
        {'r'} for arbor0 -> arbor1;
        {'l'} for arbor0 <- arbor1;
        {'l', 'r'} for arbor0 <-> arbor1;
        set() for no flow.
    """

    result = set()
    if 'b' in arbor0['neurite'] and 's' in arbor1['neurite']:
        result.add('r')
    if 's' in arbor0['neurite'] and 'b' in arbor1['neurite']:
        result.add('l')
    return result

def check_bidi_comp(cond0, cond1, arg0, arg1):
    """
    Check whether conditions are True for two arguments regardless of order.

    Parameters
    ----------
    cond0, cond1 : callable
        Function of one variable that evaluate to a bool.
    arg0, arg1 : object
        Arguments to pass to `cond0` and `cond1`.

    Returns
    -------
    result : bool
        True if `cond0` and `cond1` are respectively True for
        `arg0` and `arg1` or `arg1` and `arg0`.
    """
    
    assert callable(cond0) and callable(cond1)
    return (cond0(arg0) and cond1(arg1)) or \
        (cond1(arg0) and cond0(arg1))

def simple_overlap(regions0, regions1):
    """
    Check whether two regions in a neuropil other than EB overlap.
    """

    if regions0.intersection(regions1):
        return True
    else:
        return False

def eb_overlap(regions0, regions1):
    """
    Check whether two regions in EB overlap.
    """

    def match_tile_wedge(tile, wedge, r0, r1):
        return check_bidi_comp(lambda r: tile == r,
                               lambda r: isinstance(r, tuple) and \
                               isinstance(wedge, tuple) and len(r) >= 2 \
                               and r[0:2] == wedge[0:2],
                               r0, r1)

    # Loop through all combinations of regions:
    for r0, r1 in itertools.product(regions0, regions1):

        # Exact region match:
        if r0 == r1:
            return True
                               
        # Overlapping tile and wedge:
        if match_tile_wedge('1', ('L1', 'P'), r0, r1) or \
           match_tile_wedge('1', ('R1', 'P'), r0, r1) or \
            match_tile_wedge('2', ('R2', 'P'), r0, r1) or \
            match_tile_wedge('2', ('R3', 'P'), r0, r1) or \
            match_tile_wedge('3', ('R4', 'P'), r0, r1) or \
            match_tile_wedge('3', ('R5', 'P'), r0, r1) or \
            match_tile_wedge('4', ('R6', 'P'), r0, r1) or \
            match_tile_wedge('4', ('R7', 'P'), r0, r1) or \
            match_tile_wedge('5', ('R8', 'P'), r0, r1) or \
            match_tile_wedge('5', ('L8', 'P'), r0, r1) or \
            match_tile_wedge('6', ('L6', 'P'), r0, r1) or \
            match_tile_wedge('6', ('L7', 'P'), r0, r1) or \
            match_tile_wedge('7', ('L4', 'P'), r0, r1) or \
            match_tile_wedge('7', ('L5', 'P'), r0, r1) or \
            match_tile_wedge('8', ('L2', 'P'), r0, r1) or \
            match_tile_wedge('8', ('L3', 'P'), r0, r1):
            return True

    # No overlap found:
    return False

def check_overlap(arbor0, arbor1):
    """
    Check whether two arborizations share overlapping regions with inverse
    neurite polarities.
    """

    # Neuropils must always match for arborizations to overlap:
    if not arbor0['neuropil'] == arbor1['neuropil']:
        return set()

    # Check neuropils whose regions each have a single name:
    if arbor0['neuropil'] in ['BU', 'bu', 'FB', 'NO', 'PB']:
        if simple_overlap(arbor0['regions'], arbor1['regions']):
            return flow_direction(arbor0, arbor1)
    
    # EB needs to be treated separately because it can be divided into regions
    # that occupy the same geometric spaces:
    elif arbor0['neuropil'] == 'EB':
        if eb_overlap(arbor0['regions'], arbor1['regions']):
            return flow_direction(arbor0, arbor1)
        
    return set()
                              
if __name__ == '__main__':
    from unittest import main, TestCase

    class test_arbor_funcs(TestCase):
        def test_is_arbor(self):
            assert is_arbor({'neurite': set(['s']),
                             'neuropil': 'PB',
                             'regions': set(['R1','R2'])})

        def test_flow_direction(self):
            a = {'neurite': set(['s']),
                 'neuropil': 'PB',
                 'regions': set(['R1'])}
            b = {'neurite': set(['b']),
                 'neuropil': 'PB',
                 'regions': set(['L1'])}
            assert flow_direction(a, b) == set(['l'])

            a = {'neurite': set(['s', 'b']),
                 'neuropil': 'PB',
                 'regions': set(['R1'])}
            b = {'neurite': set(['b']),
                 'neuropil': 'PB',
                 'regions': set(['L1'])}
            assert flow_direction(a, b) == set(['l'])

        def test_check_bidi_comp(self):
            assert check_bidi_comp(lambda x: x == 1,
                                   lambda x: x == 'a',
                                   1, 'a')
            assert check_bidi_comp(lambda x: x == 1,
                                   lambda x: x == 'a',
                                   'a', 1)

        def test_simple_overlap(self):
            assert simple_overlap(set(['L1']), set(['L1', 'L2']))
                                  
            assert not simple_overlap(set(['L1']), set(['L2']))

            assert simple_overlap(set([('1', 'L1')]),
                                  set([('1', 'L1'), ('2', 'L1')]))

        def test_eb_overlap(self):
            # Two non-tile regions:
            assert eb_overlap(set([('L1', 'P', '1'), ('L1', 'M', '1')]),
                              set([('R1', 'P', '1'), ('L1', 'P', '1')]))

            # Tile that completely overlaps two corresponding specified wedges:
            assert eb_overlap(set(['1']),
                              set([('R1', 'P', '1'), ('L1', 'P', '1')]))
            
            # Tile that overlaps one specified wedge:
            assert eb_overlap(set(['1']),                              
                              set([('R1', 'P', '1')]))

            # Tile that does not overlap any specified wedges:
            assert not eb_overlap(set(['2']),
                                  set([('R1', 'P', '1'), ('L1', 'P', '1')]))

        def test_check_overlap_eb(self):
            a = {'neurite': set(['s']),
                 'neuropil': 'EB',
                 'regions': set(['1'])}
            b = {'neurite': set(['b']),
                 'neuropil': 'EB',
                 'regions': set([('L1', 'P', '1')])}
            assert check_overlap(a, b) == set(['l'])

            a = {'neurite': set(['b']),
                 'neuropil': 'EB',
                 'regions': set(['1'])}
            b = {'neurite': set(['s']),
                 'neuropil': 'EB',
                 'regions': set([('L1', 'P', '1')])}
            assert check_overlap(a, b) == set(['r'])

        def test_check_overlap_not_eb(self):
            a = {'neurite': set(['s']),
                 'neuropil': 'PB',
                 'regions': set(['L2'])}
            b = {'neurite': set(['b']),
                 'neuropil': 'PB',
                 'regions': set(['L2'])}
            assert check_overlap(a, b) == set(['l'])

            a = {'neurite': set(['b']),
                 'neuropil': 'PB',
                 'regions': set(['L2'])}
            b = {'neurite': set(['s']),
                 'neuropil': 'PB',
                 'regions': set(['L2'])}
            assert check_overlap(a, b) == set(['r'])
            
    main()
