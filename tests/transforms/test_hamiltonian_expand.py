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

import pytest
import numpy as np
import pennylane as qml
import pennylane.tape
from pennylane.interfaces.autograd import AutogradInterface
import tensorflow as tf
from pennylane.interfaces.tf import TFInterface

"""Defines the device used for all tests"""

dev = qml.device("default.qubit", wires=4)

"""Defines circuits to be used in queueing/output tests"""

with pennylane.tape.QuantumTape() as tape1:
    qml.PauliX(0)
    H1 = qml.Hamiltonian([1.5], [qml.PauliZ(0) @ qml.PauliZ(1)])
    qml.expval(H1)

with pennylane.tape.QuantumTape() as tape2:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)

    H2 = qml.Hamiltonian([1, 3, -2, 1, 1], [qml.PauliX(0) @ qml.PauliZ(2), qml.PauliZ(2), qml.PauliX(0), qml.PauliX(2), qml.PauliZ(0) @ qml.PauliX(1)])
    qml.expval(H2)

H3 = 1.5 * qml.PauliZ(0) @ qml.PauliZ(1) + 0.3 * qml.PauliX(1)

with qml.tape.QuantumTape() as tape3:
    qml.PauliX(0)
    qml.expval(H3)

TAPES = [tape1, tape2, tape3]
OUTPUTS = [-1.5, -6, -1.5]

"""Defines the data to be used for differentiation tests"""

H = [
    qml.Hamiltonian([-0.2, 0.5, 1], [qml.PauliX(1), qml.PauliZ(1) @ qml.PauliY(2), qml.PauliZ(0)])
]

GRAD_VAR = [
    np.array([0.1, 0.67, 0.3, 0.4, -0.5, 0.7])
]
TF_VAR = [
    tf.Variable([[0.1, 0.67, 0.3], [0.4, -0.5, 0.7]], dtype=tf.float64)
]

GRAD_OUT = [0.42294409781940356]
TF_OUT = [0.42294409781940356]

GRAD_OUT_2 = [[ 9.68883500e-02, -2.90832724e-01, -1.04448033e-01, -1.94289029e-09, 3.50307411e-01, -3.41123470e-01]]
TF_OUT_2 = [[ 9.68883500e-02, -2.90832724e-01, -1.04448033e-01, -1.94289029e-09, 3.50307411e-01, -3.41123470e-01]]

class TestHamiltonianExpval:
    """Tests for the hamiltonian_expand transform"""

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES, OUTPUTS))
    def test_hamiltonians(self, tape, output):
        """Tests that the hamiltonian_expand transform returns the correct value"""

        tapes, fn = qml.transforms.hamiltonian_expand(tape)
        results = dev.batch_execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval)

    def test_hamiltonian_error(self):

        with pennylane.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match=r"Passed tape must end in"):
            tapes, fn = qml.transforms.hamiltonian_expand(tape)

    @pytest.mark.parametrize(("H", "var", "output", "output2"), zip(H, GRAD_VAR, GRAD_OUT, GRAD_OUT_2))
    def test_hamiltonian_dif_autograd(self, H, var, output, output2):
        """Tests that the hamiltonian_expand tape transform is differentiable with the Autograd interface"""

        with qml.tape.JacobianTape() as tape:
            for i in range(2):
                qml.RX(np.array(0), wires=0)
                qml.RX(np.array(0), wires=1)
                qml.RX(np.array(0), wires=2)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 0])

            qml.expval(H)

        AutogradInterface.apply(tape)

        def cost(x):
            tape.set_parameters(x, trainable_only=False)
            tapes, fn = qml.transforms.hamiltonian_expand(tape)
            res = [t.execute(dev) for t in tapes]
            return fn(res)

        assert np.isclose(cost(var), output)
        assert np.allclose(qml.grad(cost)(var), output2)

    @pytest.mark.parametrize(("H", "var", "output", "output2"), zip(H, TF_VAR, TF_OUT, TF_OUT_2))
    def test_hamiltonian_dif_tensor(self, H, var, output, output2):
        """Tests that the hamiltonian_expand tape transform is differentiable with the Tensorflow interface"""

        with tf.GradientTape() as gtape:
            with qml.tape.JacobianTape() as tape:
                for i in range(2):
                    qml.RX(var[i, 0], wires=0)
                    qml.RX(var[i, 1], wires=1)
                    qml.RX(var[i, 2], wires=2)
                    qml.CNOT(wires=[0, 1])
                    qml.CNOT(wires=[1, 2])
                    qml.CNOT(wires=[2, 0])
                qml.expval(H)

            TFInterface.apply(tape)
            tapes, fn = qml.transforms.hamiltonian_expand(tape)
            res = fn([t.execute(dev) for t in tapes])

            assert np.isclose(res, output)

            g = gtape.gradient(res, var)
            assert np.allclose(list(g[0])+list(g[1]), output2)
