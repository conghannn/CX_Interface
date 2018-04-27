#!/usr/bin/env python

"""
Central Complex arborization data parser.
"""

import itertools

from parsimonious import Grammar, NodeVisitor

class NeuronArborizationParser(NodeVisitor):
    grammar = Grammar(
        """
        label = arborization (hyphen arborization)*
        arborization = neuropil slash regions slash neurite
        regions = region (bar region)*
        neuropil = "BU" / "bu" / "CRE" / "cre" / "EB" / "FB" / "IB" / "ib" / "LAL" / "lal" / "NO" / "no" / "PB" / "PS" / "ps" / "WED" / "wed"
        region = tuple2 / tuple3 / name
        tuple2 = lparen name comma name rparen
        tuple3 = lparen name comma name comma name rparen
        name = "LRB" / (side? (integer / range / alpha / list)) / (side !(integer / range / alpha / list))
        side = "LR" / "RL" / "L" / "R"
        neurite = "sb" / "bs" / "s" / "b"
        range = lbracket integer hyphen integer rbracket
        list = lbracket alpha (comma alpha)* rbracket
        integer = ~"[0-9]+"
        alpha = ~"[a-zA-Z0-9]+"
        hyphen = "-"
        bar = "|"
        slash = "/"
        comma = ","
        lparen = "("
        rparen = ")"
        lbracket = "["
        rbracket = "]"
        """)

    def visit_label(self, node, vc):
        if len(vc) > 1:
            return [vc[0]]+[c[1] for c in vc[1]]
        else:
            return vc

    def visit_arborization(self, node, vc):
        return {'neuropil': vc[0],
                'regions': set(vc[2:-2][0]),
                'neurite': vc[-1]}

    def visit_regions(self, node, vc):
        if len(vc) > 1:
            if isinstance(vc[0], list):
                tmp = vc[0]
            else:
                tmp = [vc[0]]
            return tmp+[c[1] if not isinstance(c[1], list) else c[1][0] for c in vc[1]]
        else:
            return vc

    def visit_neuropil(self, node, vc):
        return node.text

    def visit_side(self, node, vc):
        return list(node.text)

    def visit_region(self, node, vc):
        if isinstance(vc[0], list) or isinstance(vc[0], tuple):
            return vc[0]
        else:
            return node.text

    def visit_tuple2(self, node, vc):
        a = vc[1]
        b = vc[3]
        return [(x, y) for (x, y) in itertools.product(a, b)]

    def visit_tuple3(self, node, vc):
        a = vc[1]
        b = vc[3]
        c = vc[5]
        return [(x, y, z) for (x, y, z) in itertools.product(a, b, c)]

    def visit_name(self, node, vc):
        # Special case to prevent LRB from being interpreted as LB and RB:
        if node.text == 'LRB':
            return [node.text]
        tmp = []
        for c in vc[0]:
            if not c:
                continue
            elif isinstance(c[0], list):
                tmp.append(c[0])
            else:
                tmp.append(c)
        return [''.join(a) for a in itertools.product(*tmp)]

    def visit_neurite(self, node, vc):
        return set(list(node.text))

    def visit_range(self, node, vc):
        return map(str, range(int(vc[1]), int(vc[3])+1))

    def visit_list(self, node, vc):
        if isinstance(vc[1], list):
            tmp = vc[1]
        else:
            tmp = [vc[1]]
        return tmp+[c[1] if not isinstance(c[1], list) else c[1][0] for c in vc[2]]

    def visit_integer(self, node, vc):
        return node.text

    def visit_alpha(self, node, vc):
        return node.text

    def _visit_nothing(self, node, vc):
        pass

    visit_hyphen = _visit_nothing
    visit_bar = _visit_nothing
    visit_slash = _visit_nothing
    visit_comma = _visit_nothing
    visit_lparen = _visit_nothing
    visit_rparen = _visit_nothing
    visit_lbracket = _visit_nothing
    visit_rbracket = _visit_nothing

    def generic_visit(self, node, vc):
        return vc

if __name__ == '__main__':
    from unittest import main, TestCase

    class test_cx_label(TestCase):
        def setUp(self):
            self.v = NeuronArborizationParser()

        def test_single_arb_region_name_1(self):
            self.assertEqual(self.v.parse('PB/L1/s'),
                [{'neurite': set(['s']), 'neuropil': 'PB', 'regions': set(['L1'])}])

        def test_single_arb_region_name_2(self):
            self.assertEqual(self.v.parse('LAL/LHB/b'),
                [{'neurite': set(['b']), 'neuropil': 'LAL', 'regions': set(['LHB'])}])

        def test_single_arb_region_name_3(self):
            self.assertEqual(self.v.parse('cre/LRB/b'),
                [{'neurite': set(['b']), 'neuropil': 'cre', 'regions': set(['LRB'])}])

        def test_single_arb_region_name_sb(self):
            self.assertEqual(self.v.parse('PB/L1/sb'),
                [{'neurite': set(['s', 'b']), 'neuropil': 'PB', 'regions': set(['L1'])}])
            self.assertEqual(self.v.parse('PB/L1/bs'),
                [{'neurite': set(['s', 'b']), 'neuropil': 'PB', 'regions': set(['L1'])}])

        def test_single_arb_region_name_lr_1(self):
            self.assertEqual(self.v.parse('PB/LR1/b'),
                [{'neurite': set(['b']), 'neuropil': 'PB', 'regions': set(['L1', 'R1'])}])
            self.assertEqual(self.v.parse('PB/RL1/b'),
                [{'neurite': set(['b']), 'neuropil': 'PB', 'regions': set(['L1', 'R1'])}])

        def test_single_arb_region_name_lr_2(self):
            self.assertEqual(self.v.parse('PB/RL[1-2]/b'),
                             [{'neurite': set(['b']), 'neuropil': 'PB',
                               'regions': set(['L1', 'L2', 'R1', 'R2'])}])
            self.assertEqual(self.v.parse('PB/LR[1-2]/b'),
                             [{'neurite': set(['b']), 'neuropil': 'PB',
                               'regions': set(['L1', 'L2', 'R1', 'R2'])}])

        def test_single_arb_region_tuple2_1(self):
            self.assertEqual(self.v.parse('PB/(1,R1)/s'),
                [{'neurite': set(['s']), 'neuropil': 'PB', 'regions': set([('1', 'R1')])}])

        def test_single_arb_region_tuple2_2(self):
            self.assertEqual(self.v.parse('NO/(1,R)/b'),
                [{'neurite': set(['b']), 'neuropil': 'NO', 'regions': set([('1', 'R')])}])

        def test_single_arb_region_tuple3_1(self):
            self.assertEqual(self.v.parse('EB/(L7,[P,M],1)/b'),
                             [{'neurite': set(['b']), 'neuropil': 'EB',
                               'regions': set([('L7', 'P', '1'), ('L7', 'M', '1')])}])

        def test_single_arb_region_tuple3_2(self):
            self.assertEqual(self.v.parse('EB/([L1,R1],P,1)/b'),
                             [{'neurite': set(['b']), 'neuropil': 'EB',
                               'regions': set([('L1', 'P', '1'),
                                               ('R1', 'P', '1')])}])

        def test_single_arb_multiple_regions_explicit(self):
            self.assertEqual(self.v.parse('PB/L1|L2/s'),
                [{'neurite': set(['s']), 'neuropil': 'PB', 'regions': set(['L1', 'L2'])}])

        def test_single_arb_multiple_regions_range(self):
            self.assertEqual(self.v.parse('PB/L[1-2]/s'),
                [{'neurite': set(['s']), 'neuropil': 'PB', 'regions': set(['L1', 'L2'])}])

        def test_single_arb_multiple_regions_tuple_range_first(self):
            self.assertEqual(self.v.parse('FB/([3-4],L4)/s'),
                [{'neurite': set(['s']), 'neuropil': 'FB',
                  'regions': set([('3', 'L4'), ('4', 'L4')])}])

        def test_single_arb_multiple_regions_tuple_range_second(self):
            self.assertEqual(self.v.parse('PB/(1,L[1-2])/s'),
                [{'neurite': set(['s']), 'neuropil': 'PB',
                  'regions': set([('1', 'L1'), ('1', 'L2')])}])

        def test_single_arb_single_region_list(self):
            self.assertEqual(self.v.parse('PB/[L1]/s'),
                [{'neurite': set(['s']), 'neuropil': 'PB', 'regions': set(['L1'])}])

        def test_single_arb_multiple_regions_list(self):
            self.assertEqual(self.v.parse('PB/[L1,L2]/s'),
                [{'neurite': set(['s']), 'neuropil': 'PB', 'regions': set(['L1', 'L2'])}])

        def test_multiple_arbs_region_name(self):
            self.assertEqual(self.v.parse('PB/L1/s-EB/1/b'),
                [{'neurite': set(['s']), 'neuropil': 'PB',
                  'regions': set(['L1'])},
                 {'neurite': set(['b']), 'neuropil': 'EB',
                  'regions': set(['1'])}])

    main()
