import pennylane as qml
from pennylane import qaoa
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

# Defines the device and qubits

qubits = range(4)
dev = qml.device('default.qubit', wires=len(qubits))

# Defines the graph on which MaxCut is performed and the Hamiltonians

graph = nx.Graph()
graph.add_nodes_from([0, 1, 2, 3])
graph.add_edges_from([(0, 1), (1, 2), (2, 3)])

h_cost = qaoa.MaxCut(graph=graph)
h_mixer = qaoa.mixers.x_mixer(qubits)

# Defines the cost and mixer layers

cost_layer = qaoa.cost(h_cost)
mixer_layer = qaoa.mixer(h_mixer)

# Defines the QAOA circuit

depth = 1
qaoa_circuit = qaoa.circuit(cost_layer, mixer_layer, depth=depth)

# Defines the variational circuit

def circuit(params, **kwargs):

     gamma = params[0]
     alpha = params[1]

     for i in qubits:
        qml.Hadamard(wires=i)

     qaoa_circuit(gamma, alpha)

cost_function = qml.VQECost(circuit, h_cost, dev)

# Optimizes the cost function

steps = 100
optimizer = qml.AdamOptimizer()

gamma = [np.random.uniform() for i in range(0, depth)]
alpha = [np.random.uniform() for i in range(0, depth)]

params = [gamma, alpha]


for i in range(0, steps):
     params = optimizer.step(cost_function, params)

print("Final Params: {}".format(params))

# Defines the circuit from which samples are drawn

def sampling_circuit(params):

    circuit(params)
    return qml.probs(wires=qubits)

qnode = qml.QNode(sampling_circuit, dev)
res = qnode(params)

plt.bar(range(2**len(qubits)), res)
plt.show()