import pennylane as qml
from pennylane.templates import TimeEvolution
import numpy as np

n_wires = 2
dev = qml.device('default.qubit', wires=n_wires)

hamiltonian = qml.Hamiltonian([1, 1], [qml.PauliX(0) @ qml.Hermitian(np.array([[1, 1], [1, 1]]), 1), qml.PauliX(0)])

@qml.qnode(dev)
def circuit():
    TimeEvolution(hamiltonian, 2, N=3)
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

print(circuit())
print(circuit.draw())