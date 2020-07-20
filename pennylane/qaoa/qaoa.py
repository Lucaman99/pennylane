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
This file contains built-in functions for defining QAOA workflows
"""
from collections.abc import Iterable
import networkx
import pennylane as qml
from pennylane.wires import Wires


def _check_iterable_graph(graph):
    """ Checks if a graph supplied in 'Iterable format' is formatted correctly

        Args:
            graph (Iterable): The graph that is being checked
    """

    for g in graph:

        if not isinstance(g, Iterable):
            raise ValueError(
                "Elements of `graph` must be Iterable objects, got {}".format(type(g).__name__)
            )
        if len(g) != 2:
            raise ValueError(
                "Elements of `graph` must be Iterable objects of length 2, got length {}".format(
                    len(g)
                )
            )
        if g[0] == g[1]:
            raise ValueError("Edges must end in distinct nodes, got {}".format(g))

    if len({tuple(g) for g in graph}) != len(graph):
        raise ValueError("Nodes cannot be connected by more than one edge")


##################### Mixer Hamiltonians #####################


def x_mixer(wires):
    r""""Creates the basic Pauli-X mixer Hamiltonian used in the original `QAOA paper <https://arxiv.org/abs/1411.4028>`__.
    This Hamiltonian is defined as:

    .. math:: H_M \ = \ \displaystyle\sum_{i} X_{i},

    where :math:`i` ranges over all qubits, and :math:`X_i` denotes the Pauli-X operator on the :math:`i`-th qubit (wire).
    Args:
        wires (Iterable or Wires): The collection of wires to which the observables in the Hamiltonian correspond.
    Returns:
        ``qml.Hamiltonian`` object encoding the Hamiltonian
    """

    ##############
    # Input checks

    wires = Wires(wires)

    coeffs = [1 for w in wires]
    obs = [qml.PauliX(w) for w in wires]

    return qml.Hamiltonian(coeffs, obs)


def xy_mixer(graph):
    r""""Creates the generalized SWAP/XY mixer outlined in `this paper <https://arxiv.org/abs/1709.03489>`__, defined
        as:

        .. math:: H_M \ = \ \frac{1}{2} \displaystyle\sum_{(i, j) \in E(G)} X_i X_j \ + \ Y_i Y_j,

        for some graph :math:`G`. :math:`X_i` and :math:`Y_i` denote the Pauli-X and Pauli-Y operators on the :math:`i`-th
        qubit respectively.
        Args:
            graph (Iterable or networkx.Graph) A graph defining the pairs of qubits (wires) on which each term of the Hamiltonian acts.
        Returns:
            ``qml.Hamiltonian`` object encoding the Hamiltonian
        """

    ##############
    # Input checks

    if isinstance(graph, networkx.Graph):
        graph = graph.edges

    elif isinstance(graph, Iterable):
        _check_iterable_graph(graph)

    else:
        raise ValueError(
            "Input graph must be a networkx.Graph object or Iterable, got {}".format(
                type(graph).__name__
            )
        )

    coeffs = 2 * [0.5 for g in graph]

    obs = []
    for g in graph:
        obs.append(qml.PauliX(Wires(g[0])) @ qml.PauliX(Wires(g[1])))
        obs.append(qml.PauliY(Wires(g[0])) @ qml.PauliY(Wires(g[1])))

    return qml.Hamiltonian(coeffs, obs)
