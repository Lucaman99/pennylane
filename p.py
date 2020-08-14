import pennylane as qml
from pennylane import qaoa
from networkx import Graph

# Defines the wires and the graph on which MaxCut is being performed

wires = range(3)
graph = Graph([(0, 1), (1, 2), (2, 0)])

# Defines the QAOA cost and mixer Hamiltonians

cost_h, mixer_h = qaoa.maxcut(graph)

# Defines a layer of the QAOA ansatz, from the cost and mixer Hamiltonians

def qaoa_layer(gamma, alpha):
    qaoa.cost_layer(gamma, cost_h)
    qaoa.mixer_layer(alpha, mixer_h)

# Repeatedly applies layers of the QAOA ansatz

def circuit(params, **kwargs):

    for w in wires:
        qml.Hadamard(wires=w)

    qml.layer(qaoa_layer, 2, params[0], params[1])

# Defines the device and the QAOA cost function

dev = qml.device('default.qubit', wires=len(wires))
cost_function = qml.VQECost(circuit, cost_h, dev)

print(cost_function([[1, 1], [1, 1]]))
