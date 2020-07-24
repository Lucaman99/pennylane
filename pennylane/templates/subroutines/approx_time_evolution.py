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
Contains the ``TimeEvolution`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.wires import Wires


@template
def ApproxTimeEvolution(hamiltonian, time, n, wires):
    r""" Applies the Trotterized time-evolution operator for an arbitrary Hamiltonian, expressed in terms
    of Pauli gates. The general
    time-evolution operator for a time-independent Hamiltonian is given by:

    .. math:: U(t) \ = \ e^{-i H t},

    for some Hamiltonian of the form:

    .. math:: H \ = \ \displaystyle\sum_{j} H_j.

    In general, implementing this unitary with a set of quantum gates is difficult, as the terms :math:`H_j` don't
    necessarily commute with one another. However, we are able to exploit the Trotter-Suzuki decomposition formula:

    .. math:: e^{A \ + \ B} \ = \ \lim_{n \to \infty} \Big[ e^{A/n} e^{B/n} \Big]^n

    to implement an approximation of the time-evolution operator as:

    .. math:: U \ \approx \ \displaystyle\prod_{k \ = \ 1}^{n} \displaystyle\prod_{j} e^{-i H_j t / n},

    with the approximation becoming better for larger :math:`n`. It is also important to note that
    this decomposition is exact for any value of :math:`n` when each term of the Hamiltonian, :math:`H_n`,
    commutes with every other term.

    .. note::

       This template uses the ``PauliRot`` operation in order to implement
       exponentiated terms of the input Hamiltonian. This operation only takes
       terms that are explicitly written in terms of products of Pauli matrices (``PauliX``, ``PauliY``, ``PauliZ``, and ``Identity``).
       Thus, each term in the Hamiltonian must be expressed this way upon input, or else an error will be raised.

    Args:
        hamiltonian (pennylane.Hamiltonian): The Hamiltonian defining the
           time-evolution operator. The indices of the observables in the Hamiltonian correspond to the index of
           the ``wires`` element to which each observable is being applied.
           The Hamiltonian must be explicitly written
           in terms of products of Pauli gates (X, Y, Z, and I).

        time (int or float): The time of evolution, namely the parameter :math:`t` in :math:`e^{- i H t}`.

        n (int): The number of Trotter steps used when approximating the time-evolution operator.

        wires (Iterable or Wires): The wires on which the template is applied.

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        The template is used inside a qnode:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import ApproxTimeEvolution

            n_wires = 2
            wires = range(n_wires)

            dev = qml.device('default.qubit', wires=n_wires)

            coeffs = [1, 1]
            obs = [qml.PauliX(0), qml.PauliX(1)]
            hamiltonian = qml.Hamiltonian(coeffs, obs)

            @qml.qnode(dev)
            def circuit(time):
                TimeEvolution(hamiltonian, wires, time, n=1)
                return [qml.expval(qml.PauliZ(wires=i)) for i in wires]

        >>> circuit(1)
        [-0.41614684 -0.41614684]
    """

    pauli = {"Identity": "I", "PauliX": "X", "PauliY": "Y", "PauliZ": "Z"}

    ###############
    # Input checks

    wires = Wires(wires)

    if not isinstance(hamiltonian, qml.vqe.vqe.Hamiltonian):
        raise ValueError(
            "`hamiltonian` must be of type pennylane.Hamiltonian, got {}".format(
                type(hamiltonian).__name__
            )
        )

    if not isinstance(n, (int, qml.variable.Variable)):
        raise ValueError("`N` must be of type int, got {}".format(type(n).__name__))

    if not isinstance(time, (int, float, qml.variable.Variable)):
        raise ValueError("`time` must be of type int or float, got {}".format(type(time).__name__))

    ###############

    theta = []
    pauli_words = []
    wire_index = []

    for i, term in enumerate(hamiltonian.ops):

        prod = (-2 * time * hamiltonian.coeffs[i]) / n
        word = ""

        try:
            if isinstance(term.name, str):
                word = pauli[term.name]

            if isinstance(term.name, list):
                for j in term.name:
                    word += pauli[j]

        except KeyError as error:
            raise ValueError(
                "`hamiltonian` must be written in terms of Pauli matrices, got {}".format(error)
            )

        count = 0
        for j in list(word):
            if j == "I":
                count += 1

        if count != len(word):

            theta.append(prod)
            pauli_words.append(word)
            wire_index.append(term.wires)

    for i in range(n):

        for j, term in enumerate(pauli_words):
            qml.PauliRot(theta[j], term, wires=[wires[i] for i in wire_index[j]])