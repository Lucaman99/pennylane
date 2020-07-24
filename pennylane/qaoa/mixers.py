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
This file contains built-in functions for constructing QAOA mixer Hamiltonians.
"""
import networkx as nx
import pennylane as qml


def x_mixer(n):
    r"""Creates a basic Pauli-X mixer Hamiltonian.

    This Hamiltonian is defined as:

    .. math:: H_M \ = \ \displaystyle\sum_{i} X_{i},

    where :math:`i` ranges over all wires, and :math:`X_i`
    denotes the Pauli-X operator on the :math:`i`-th wire.
    This is the mixer that was outlined in the `original QAOA
    paper <https://arxiv.org/abs/1411.4028>`__.


    Args:
        n (int): The number of wires on which the Hamiltonian is applied

    Returns:
        Hamiltonian:
    """

    wires = range(n)

    coeffs = [1 for w in wires]
    obs = [qml.PauliX(w) for w in wires]

    return qml.Hamiltonian(coeffs, obs)


def xy_mixer(graph):
    r"""Creates a generalized SWAP/XY mixer Hamiltonian.

    This mixer Hamiltonian is defined as:

    .. math:: H_M \ = \ \frac{1}{2} \displaystyle\sum_{(i, j) \in E(G)} X_i X_j \ + \ Y_i Y_j,

    for some graph :math:`G`. :math:`X_i` and :math:`Y_i` denote the Pauli-X and Pauli-Y operators on the :math:`i`-th
    wire respectively. This mixer was first presented in `this paper <https://arxiv.org/abs/1709.03489>`__.

    Args:
        graph (nx.Graph): A graph defining the pairs of wires on which each term of the Hamiltonian acts.

    Returns:
         Hamiltonian:
        """

    if not isinstance(graph, nx.Graph)::
        raise ValueError(
            "Input graph must be a nx.Graph object, got {}".format(type(graph).__name__)
        )

    edges = graph.edges
    coeffs = 2 * [0.5 for e in edges]

    obs = []
    for node1, node2 in edges:
        obs.append(qml.PauliX(node1) @ qml.PauliX(node2))
        obs.append(qml.PauliY(node1) @ qml.PauliY(node2))

    return qml.Hamiltonian(coeffs, obs)
