# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Methods for generating QAOA cost Hamiltonians corresponding to
different optimization problems.
"""

import networkx as nx
import pennylane as qml
from pennylane import qaoa

########################
# Hamiltonian components

def pauli_driver(wires, state):

    if state == 0:
        coeffs = [-1 for _ in wires]
    elif state == 1:
        coeffs = [1 for _ in wires]
    else:
        raise ValueError("'state' argument must be either 0 or 1, got {}".format(state))

    ops = [qml.PauliZ(w) for w in wires]
    return qml.Hamiltonian(coeffs, ops)

def edge_driver(graph, reward):

    if not isinstance(graph, nx.Graph):
        raise ValueError(
            "Input graph must be a nx.Graph object, got {}".format(type(graph).__name__)
        )

    reward = list(set(reward) - {'01'})
    sign = -1

    if len(reward) == 2:
        reward = list({'00', '10', '11'} - set(reward))[0]
        sign = 1

    coeffs = []
    ops = []

    if reward == '00':
        for e in graph.edges:
            coeffs.extend([0.5*sign, 0.5*sign, 0.5*sign])
            ops.extend([qml.PauliZ(e[0]) @ qml.PauliZ(e[1]), qml.PauliZ(e[0]), qml.PauliZ(e[1])])

    if reward == '10':
        for e in graph.edges:
            coeffs.append(-1*sign)
            ops.append(qml.PauliZ(e[0]) @ qml.PauliZ(e[1]))

    if reward == '11':
        for e in graph.edges:
            coeffs.extend([0.5*sign, -0.5*sign, -0.5*sign])
            ops.extend([qml.PauliZ(e[0]) @ qml.PauliZ(e[1]), qml.PauliZ(e[0]), qml.PauliZ(e[1])])

    return qml.Hamiltonian(coeffs, ops)

#######################
# Optimization problems

def maxcut(graph):
    r"""Returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the
    MaxCut problem, for a given graph.

    The goal of the MaxCut problem for a particular graph is to find a partition of nodes into two sets,
    such that the number of edges in the graph with endpoints in different sets is maximized. Formally,
    we wish to find the `cut of the graph <https://en.wikipedia.org/wiki/Cut_(graph_theory)>`__ such
    that the number of edges crossing the cut is maximized.

    The MaxCut cost Hamiltonian is defined as:

    .. math:: H_C \ = \ \frac{1}{2} \displaystyle\sum_{(i, j) \in E(G)} \big( Z_i Z_j \ - \ \mathbb{I} \big),

    where :math:`G` is a graph, :math:`\mathbb{I}` is the identity, and :math:`Z_i` and :math:`Z_j` are
    the Pauli-Z operators on the :math:`i`-th and :math:`j`-th wire respectively.

    The mixer Hamiltonian returned from :func:`~qaoa.maxcut` is :func:`~qaoa.x_mixer` applied to all wires.

    .. note::

        **Recommended initialization circuit:**
            Even superposition over all basis states

    Args:
        graph (nx.Graph): a graph defining the pairs of wires on which each term of the Hamiltonian acts

    Returns:
        (.Hamiltonian, .Hamiltonian):

    **Example**

    >>> graph = nx.Graph([(0, 1), (1, 2)])
    >>> cost_h, mixer_h = qml.qaoa.maxcut(graph)
    >>> print(cost_h)
    >>> print(mixer_h)
    (-0.5) [I0 I1] + (0.5) [Z0 Z1] + (-0.5) [I1 I2] + (0.5) [Z1 Z2]
    (1.0) [X0] + (1.0) [X1] + (1.0) [X2]
    """

    return (edge_driver(graph, ['10', '01']), qaoa.x_mixer(graph.nodes))

def max_independent_set(graph, constrained=True):
    r"""Returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the MaxIndependentSet problem,
    for a given graph.


    """

    if constrained:
        return (pauli_driver(graph.nodes, 1), qaoa.bit_flip_mixer(graph, 0))
    else:
        cost_h = edge_driver(graph, ['10', '01', '00']) + pauli_driver(graph.nodes, 1)
        mixer_h = qaoa.x_mixer(graph.nodes)

        return (cost_h, mixer_h)

def min_vertex_cover(graph, constrained=True):
    r"""Returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the Minimum Vertex Cover problem,
    for a given graph.

    The goal of the Minimum Vertex Cover problem is to find the smallest
    `vertex cover <https://en.wikipedia.org/wiki/Vertex_cover>`__ of a graph (a collection of nodes such that
    every edge in the graph has one of the nodes as an endpoint).

    The Minimum Vertex Cover cost Hamiltonian is defined as:

    .. math:: H_C \ = \ \frac{|V(G)|}{2} \displaystyle\sum_{(i, j) \in E(G)} \big(Z_i Z_j \ - \ \mathbb{I} \big) \ - \ \displaystyle\sum_{j} Z_j

    where :math:`G` is a graph, :math:`\mathbb{I}` is the identity, and :math:`Z_i` and :math:`Z_j` are
    the Pauli-Z operators on the :math:`i`-th and :math:`j`-th wire respectively.

    The mixer Hamiltonian returned from :func:`~qaoa.maxcut` is :func:`~qaoa.x_mixer` applied to all wires.

    .. note::

        **Recommended initialization circuit:**
            Even superposition over all basis states

    Args:
        graph (nx.Graph): a graph defining the pairs of wires on which each term of the Hamiltonian acts

    Returns:
        (.Hamiltonian, .Hamiltonian):

    **Example**

    >>> graph = nx.Graph([(0, 1), (1, 2)])
    >>> cost_h, mixer_h = qml.qaoa.min_vertex_cover(graph)
    >>> print(cost_h)
    >>> print(mixer_h)
    (-1.5) [I0 I1] + (1.5) [Z0 Z1] + (-1.5) [I1 I2] + (1.5) [Z1 Z2] + (-1.0) [Z0] + (-1.0) [Z1] + (-1.0)[Z2]
    (1.0) [X0] + (1.0) [X1] + (1.0) [X2]
    """

    if constrained:
        return (pauli_driver(graph.nodes, 0), qaoa.bit_flip_mixer(graph, 1))
    else:
        cost_h = edge_driver(graph, ['11', '10', '01']) + 0.5*pauli_driver(graph.nodes, 0)
        mixer_h = qaoa.x_mixer(graph.nodes)

        return (cost_h, mixer_h)

def maxclique(graph, constrained=True):
    r"""Returns the QAOA cost Hamiltonian and the reccommended mixer corresponding to the MaxClique problem,
    for a given graph.

    The goal of MaxClique is to find the largest `clique <https://en.wikipedia.org/wiki/Clique_(graph_theory)>`__ of a
    graph (the largest subgraph with all nodes sharing an edge).

    The MaxClique cost Hamiltonian is defined as:

    .. math:: H_C \ = \ \frac{1}{2} \frac{(i, j) \in E(\bar{G})} (Z_i Z_j \ - \ Z_i \ - \ Z_j) \ + \ \displaystyle\sum_{j} Z_i


    """

    if constrained:
        return (pauli_driver(graph.nodes, 1), qaoa.bit_flip_mixer(nx.complement(graph), 0))
    else:
        cost_h = edge_driver(nx.complement(graph), ['10', '01', '00']) + 0.5*pauli_driver(graph.nodes, 1)
        mixer_h = qaoa.x_mixer(graph.nodes)

        return (cost_h, mixer_h)
