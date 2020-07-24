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

import pennylane as qml
from pennylane.templates import ApproxTimeEvolution
from pennylane.operation import Tensor


def _diagonal_terms(hamiltonian):

    val = True

    for i in hamiltonian.ops:
        i = Tensor(i) if isinstance(i.name, str) else i
        for j in i.obs:
            if j.name != "PauliZ" and j.name != "Identity":
                val = False

    return val


def cost_layer(hamiltonian):

    if not isinstance(hamiltonian, qml.Hamiltonian):
        raise ValueError(
            "hamiltonian must be of type pennylane.Hamiltonian, got {}".format(
                type(hamiltonian).__name__
            )
        )

    if not _diagonal_terms(hamiltonian):
        raise ValueError("hamiltonian must be written only in terms of PauliZ and Identity gates")

    return lambda gamma, wires: ApproxTimeEvolution(hamiltonian, gamma, 1, wires)


def mixer_layer(hamiltonian):

    if not isinstance(hamiltonian, qml.Hamiltonian):
        raise ValueError(
            "hamiltonian must be of type pennylane.Hamiltonian, got {}".format(
                type(hamiltonian).__name__
            )
        )

    return lambda alpha, wires: ApproxTimeEvolution(hamiltonian, alpha, 1, wires)
