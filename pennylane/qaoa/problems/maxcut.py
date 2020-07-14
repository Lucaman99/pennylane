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
This file contains the function that builds the QAOA MaxCut Hamiltonian
"""
import pennylane as qml

def MaxCut(graph):

    ###############
    # Input checks

    ###############

    coeffs = []
    obs = []

    for e in graph.edges:

        #obs.append(qml.Identity(e[0]) @ qml.Identity(e[1]))
        #coeffs.append(0.5)

        obs.append(qml.PauliZ(e[0]) @ qml.PauliZ(e[1]))
        coeffs.append(1)

    return qml.Hamiltonian(coeffs, obs)