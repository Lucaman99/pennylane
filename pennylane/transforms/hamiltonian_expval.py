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
"""
Contains the hamiltonian expectation value tape transform
"""
import itertools
import numpy as np
import pennylane as qml


def hamiltonian_expval(tape):
    r"""
    Returns a list of tapes, and a classical processing function, for computing the expectation
    value of a Hamiltonian.

    Args:
        tape (.QuantumTape) the tape used when calculating the expectation value
        of the Hamiltonian.

    Returns:
        tuple[list[.QuantumTape], func]: Returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape results to compute the expectation value.

    **Example**

    Given a tape of the form,

    .. code-block:: python3

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)

            H = qml.PauliY(2) @ qml.PauliZ(1) + 0.5 * qml.PauliZ(2) + qml.PauliZ(1)
            qml.expval(H)

    We can use the ``hamiltonian_expval`` transform to generate new tapes and a classical
    post-processing function for computing the expectation value of the Hamiltonian.

    >>> tapes, fn = qml.tape.transforms.hamiltonian_expval(tape)

    We can evaluate these tapes on a device:

    >>> dev = qml.device("default.qubit", wires=3)
    >>> res = dev.batch_execute(tapes)

    Applying the processing function results in the expectation value of the Hamiltonian:

    >>> fn(res)
    -0.5
    """

    hamiltonian = tape.measurements[0].obs

    if not isinstance(hamiltonian, qml.Hamiltonian) or len(tape.measurements) > 1:
        raise ValueError(
            "Passed tape must end in `qml.expval(H)`, where H is of type `qml.Hamiltonian`"
        )

    hamiltonian.simplify()
    return qml.transforms.measurement_grouping(tape, H.ops, H.coeffs)
