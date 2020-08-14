import pennylane as qml
from pennylane import qaoa
from networkx import Graph
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import numpy as np

# Defines the wires and the graph on which MaxCut is being performed

wires = range(4)
graph = Graph([(0, 1), (1, 2), (2, 0), (2, 3)])

# Defines the QAOA cost and mixer Hamiltonians

cost_h, mixer_h = qaoa.maxclique(graph)

# Defines a layer of the QAOA ansatz, from the cost and mixer Hamiltonians

def qaoa_layer(gamma, alpha):
    qaoa.cost_layer(gamma, cost_h)
    qaoa.mixer_layer(alpha, mixer_h)

# Repeatedly applies layers of the QAOA ansatz

def circuit(params, **kwargs):
    
    params = [params[0:2], params[2:4]]

    for w in wires:
        qml.Hadamard(wires=w)
    
    qml.layer(qaoa_layer, 2, params[0], params[1])
    print("Done")

# Defines the device and the QAOA cost function

dev = qml.device('qulacs.simulator', wires=len(wires))
cost_function = qml.VQECost(circuit, cost_h, dev)

# Creates the optimizer

#optimizer = qml.AdagradOptimizer()
#steps = 50
params = [
        [np.random.randint(0, 100)/100 for i in range(2)],
        [np.random.randint(0, 100)/100 for i in range(2)]
        ]

#for i in range(steps):
#    params = optimizer.step(cost_function, params)
#    print(i)

opt = minimize(cost_function, x0=params, method='COBYLA', options={'maxiter':50})
params = opt['x']
print(opt)
print(params)

@qml.qnode(dev)
def dist_circuit(gamma=None, alpha=None):
    circuit(gamma+alpha)
    return qml.probs(wires=wires)

output = dist_circuit(gamma=list(params[0]), alpha=list(params[1]))

plt.bar(range(2**len(wires)), output)
plt.show()

