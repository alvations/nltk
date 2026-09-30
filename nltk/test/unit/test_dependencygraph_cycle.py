# Natural Language Toolkit: DependencyGraph cycle detection through mapped deps
#
# Copyright (C) 2001-2026 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
"""Every graph the parsers build stores ``deps`` as a relation to addresses
mapping. Iterating that mapping yields relation labels, which the membership
guard drops, so ``contains_cycle`` never saw a cycle built through CoNLL
parsing or ``add_arc``. These tests drive the real reader and the real MST
parser on such graphs; nothing is mocked."""

import gc
import warnings

import pytest

from nltk.parse.dependencygraph import DependencyGraph, _dependent_addresses
from nltk.test.unit.test_quadratic_dos import _assert_subquadratic

# (label, Malt-TAB or CoNLL text, cycle contains_cycle must report)
_CONLL_CYCLES = [
    ("two-cycle", "a N 0 ROOT\nb N 3 dep\nc N 2 dep\n", [2, 3]),
    ("self-loop", "a N 0 ROOT\nb N 2 dep\n", [2]),
    (
        "ring of five",
        "r N 0 ROOT\na N 6 d\nb N 2 d\nc N 3 d\nd N 4 d\ne N 5 d\n",
        [2, 3, 4, 5, 6],
    ),
    ("cycle off the root", "a N 0 ROOT\nb N 1 d\nc N 4 d\nd N 3 d\n", [3, 4]),
    (
        "ten-column two-cycle",
        "1\ta\ta\tN\tN\t_\t0\tROOT\t_\t_\n"
        "2\tb\tb\tN\tN\t_\t3\tdep\t_\t_\n"
        "3\tc\tc\tN\tN\t_\t2\tdep\t_\t_\n",
        [2, 3],
    ),
]


def _graph(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a cyclic graph has no root dependent
        return DependencyGraph(text)


def test_contains_cycle_follows_relation_mapped_dependencies():
    graph = DependencyGraph("a N 0 ROOT\n" "b N 3 dep\n" "c N 2 dep\n")

    assert graph.contains_cycle() == [2, 3]


@pytest.mark.parametrize(
    "label, text, cycle", _CONLL_CYCLES, ids=[c[0] for c in _CONLL_CYCLES]
)
def test_every_cycle_built_through_parsing_is_found(label, text, cycle):
    graph = _graph(text)
    assert isinstance(graph.nodes[cycle[0]]["deps"], dict), "parsing stores a mapping"
    before = sorted(graph.nodes)
    assert graph.contains_cycle() == cycle
    assert graph.get_cycle_path(graph.nodes[cycle[0]], cycle[0]) == cycle
    assert sorted(graph.nodes) == before, "no node was materialised"


def test_a_cycle_built_with_add_arc_is_found():
    graph = DependencyGraph()
    for address, word in ((1, "a"), (2, "b")):
        graph.add_node({"address": address, "word": word, "deps": {}, "rel": "x"})
    graph.add_arc(1, 2)
    graph.add_arc(2, 1)
    assert graph.contains_cycle() == [1, 2]


@pytest.mark.parametrize(
    "text",
    [
        "a N 0 ROOT\nb N 1 d\nc N 2 d\n",  # a chain
        "a N 0 ROOT\nb N 99 d\n",  # a dangling head is not a cycle
        "a N 0 ROOT\nb N 1 d\nc N 1 d\nd N 1 d\n",  # a star
    ],
)
def test_acyclic_graphs_stay_acyclic(text):
    graph = _graph(text)
    before = sorted(graph.nodes)
    assert graph.contains_cycle() is False
    assert sorted(graph.nodes) == before


def test_the_legacy_list_shape_still_works():
    # the shape the contains_cycle doctest uses
    graph = DependencyGraph()
    graph.nodes = {
        0: {"word": None, "deps": [1], "rel": "TOP", "address": 0},
        1: {"word": None, "deps": [2], "rel": "NTOP", "address": 1},
        2: {"word": None, "deps": [4], "rel": "NTOP", "address": 2},
        3: {"word": None, "deps": [1], "rel": "NTOP", "address": 3},
        4: {"word": None, "deps": [3], "rel": "NTOP", "address": 4},
    }
    assert graph.contains_cycle() == [1, 2, 4, 3]
    assert graph.get_cycle_path(graph.nodes[1], 1) == [1, 2, 4, 3]
    assert list(_dependent_addresses(graph.nodes[2])) == [4]


def test_dependent_addresses_reads_values_not_labels():
    node = {"deps": {"nsubj": [2], "obj": [3, 4]}}
    assert sorted(_dependent_addresses(node)) == [2, 3, 4]
    assert list(_dependent_addresses({"deps": [5]})) == [5]
    assert list(_dependent_addresses({"deps": {}})) == []


def test_redirect_arcs_keeps_the_mapping_shape():
    graph = DependencyGraph("a N 0 ROOT\nb N 1 x\nc N 1 y\n")
    graph.redirect_arcs([3], 2)
    assert dict(graph.nodes[1]["deps"]) == {"x": [2], "y": [2]}
    legacy = DependencyGraph()
    legacy.nodes[1]["deps"] = [2, 3]
    legacy.redirect_arcs([3], 2)
    assert legacy.nodes[1]["deps"] == [2, 2]


def test_a_non_integer_head_is_refused():
    with pytest.raises(ValueError):
        DependencyGraph("a N 0 ROOT\nb N x dep\n")


def _ring(n):
    lines = ["r N 0 ROOT"] + [
        "w%d N %d d" % (i, i + 1 if i < n else 2) for i in range(2, n + 1)
    ]
    return "\n".join(lines) + "\n"


def _chain(n):
    return (
        "\n".join("w%d N %d d" % (i, i + 1 if i < n else 0) for i in range(1, n + 1))
        + "\n"
    )


def test_large_cycles_and_chains_are_found_iteratively():
    ring = _graph(_ring(50_000))
    assert len(ring.contains_cycle()) == 49_999
    chain = _graph(_chain(50_000))
    assert chain.contains_cycle() is False  # no RecursionError on a 50k deep chain


def test_cycle_detection_is_linear():
    graphs = {n: _graph(_chain(n)) for n in (25_000, 100_000)}

    def op(n):
        # the collector otherwise rescans the live DFS stack objects on every
        # generation and hides the linear scaling behind its own cost
        gc.disable()
        try:
            graphs[n].contains_cycle()
        finally:
            gc.enable()

    _assert_subquadratic(op, 25_000, 100_000)


# The MST parser is the one library caller: its cycle collapse used to be dead.
def _mst_parse(matrix, tokens):
    from nltk.parse.nonprojectivedependencyparser import (
        DependencyScorerI,
        ProbabilisticNonprojectiveParser,
    )

    class Scorer(DependencyScorerI):
        def train(self, graphs):
            pass

        def score(self, graph):
            return matrix

    parser = ProbabilisticNonprojectiveParser()
    parser.train([], Scorer())
    (graph,) = list(parser.parse(tokens, [None] * len(tokens)))
    return {
        address: sorted(_dependent_addresses(node))
        for address, node in sorted(graph.nodes.items())
    }


def test_the_mst_parser_now_returns_a_rooted_tree_for_its_docstring_scorer():
    # best incoming arcs make v1 and v2 each other's head; before the fix the
    # cycle was never collapsed and the parse came back rootless and cyclic
    matrix = [
        [[], [5], [1], [1]],
        [[], [], [11], [4]],
        [[], [10], [], [5]],
        [[], [8], [8], []],
    ]
    assert _mst_parse(matrix, ["v1", "v2", "v3"]) == {0: [1], 1: [2], 2: [3], 3: []}


def test_the_mst_parser_collapses_a_ring_and_two_separate_cycles():
    ring = [
        [[], [1], [1], [1]],
        [[], [], [9], [1]],
        [[], [1], [], [9]],
        [[], [9], [1], []],
    ]
    assert _mst_parse(ring, ["a", "b", "c"]) == {0: [1], 1: [2], 2: [3], 3: []}
    two = [
        [[], [1], [1], [1], [1]],
        [[], [], [9], [1], [1]],
        [[], [9], [], [1], [1]],
        [[], [1], [1], [], [9]],
        [[], [1], [1], [9], []],
    ]
    assert _mst_parse(two, ["a", "b", "c", "d"]) == {
        0: [1, 3],
        1: [2],
        2: [],
        3: [4],
        4: [],
    }
