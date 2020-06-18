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
Contains the ``CustomEntanglerLayers`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane.templates.decorator import template
from pennylane.ops import CNOT, RX, CRX
from pennylane.templates import broadcast
from pennylane.templates.utils import (
    check_shape,
    check_no_variable,
    check_number_of_layers,
    get_shape,
    check_shapes,
)
from pennylane.wires import Wires


@template
def CustomEntanglerLayers(
    rotation_weights, wires, rotation=None, coupling=None, coupling_weights=None, pattern=None
):
    r"""Layers consisting of one-parameter single-qubit rotations on each qubit, followed by a sequence of
    double-qubit gates (parametrized or non-parametrized)

    The placement of double-qubit gates on the circuit is determined by a user-passed
    `pattern`.

    .. figure:: ../../_static/templates/layers/basic_entangler.png
        :align: center
        :width: 40%
        :target: javascript:void(0);

    The number of layers :math:`L` is determined by the first dimension of the
    first argument ``rotation_weights``.
    When using a single wire, the template only applies the single
    qubit gates in each layer.

    .. note::

        This template follows the convention of dropping the entanglement between the last and the first
        qubit when using only two wires, so the entangler is not repeated on the same wires.
        In this case, only one two-qubit gate is applied in each layer:

        .. figure:: ../../_static/templates/layers/basic_entangler_2wires.png
            :align: center
            :width: 30%
            :target: javascript:void(0);

    Args:
        rotation_weights (array[float]): array of weights with shape ``(L, len(wires))``, each weight is used as a parameter
                                for the rotation
        coupling_weights (array[float]): array of weights with shape ``(L, len(OPTION))``, each weight is used as a parameter
                                for the coupling
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers or strings, or
            a Wires object.
        rotation (pennylane.ops.Operation): one-parameter single-qubit gate to use,
                                            if ``None``, :class:`~pennylane.ops.RX` is used as default
        pattern (str or "Custom" list): A keyword that determines how the double-qubit gates will be placed on the 
                                        circuit. `pattern='ring'` is used as default.
    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        The template is used inside a qnode:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import BasicEntanglerLayers
            from math import pi

            n_wires = 3
            dev = qml.device('default.qubit', wires=n_wires)

            @qml.qnode(dev)
            def circuit(rotation_weights, coupling_weights):
                CustomEntanglerLayers(rotation_weights=rotation_weights, 
                                      coupling_weights=coupling_weights, 
                                      wires=range(n_wires),
                                      rotation=qml.RX,
                                      coupling=qml.CRX,
                                      pattern=[[0, 1]])
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        >>> circuit([[pi, pi, pi]])
        [-1., 1., -1.]
    """

    #############
    # Input checks

    n_parameters = {
        "single": len(wires),
        "double": 0 if len(wires) in [0, 1] else len(wires) // 2,
        "double_odd": 0 if len(wires) in [0, 1] else (len(wires) - 1) // 2,
        "chain": 0 if len(wires) in [0, 1] else len(wires) - 1,
        "ring": 0 if len(wires) in [0, 1] else (1 if len(wires) == 2 else len(wires)),
        "pyramid": 0 if len(wires) in [0, 1] else sum(i + 1 for i in range(len(wires) // 2)),
        "all_to_all": 0 if len(wires) in [0, 1] else len(wires) * (len(wires) - 1) // 2,
    }

    OPTIONS = ["single", "double", "double_odd", "chain", "ring", "pyramid", "all_to_all"]

    if rotation is None:
        rotation = RX

    wires = Wires(wires)

    check_no_variable(rotation, msg="'rotation' cannot be differentiable")

    repeat = check_number_of_layers([rotation_weights])

    expected_shape = (repeat, len(wires))
    check_shape(
        rotation_weights,
        expected_shape,
        msg="'rotation_weights' must be of shape {}; got {}"
        "".format(expected_shape, get_shape(rotation_weights)),
    )

    if pattern is None:
        pattern = "ring"

    if coupling is None and coupling_weights is None:
        coupling = CNOT
    
    if coupling is None and coupling_weights is not None:
        coupling = CRX

    if coupling.num_wires != 2:
        raise ValueError(
            "`coupling` accepts 2-wire gates, instead got {} wire(s)".format(coupling.num_wires)
        )

    if coupling.num_params == 0 and coupling_weights is not None:
        raise ValueError("Gate '{}' does not take parameters".format(coupling))

    if coupling.num_params != 0 and coupling_weights is None:
        raise ValueError("Gate '{}' must take parameters".format(coupling))

    if isinstance(pattern, list):
        check_shapes(pattern, [(2,)], msg="Elements of custom 'pattern' must be of shape (2,)")

    if coupling_weights is not None:

        repeat_coupling = check_number_of_layers([coupling_weights])
        if repeat_coupling != repeat:
            raise ValueError(
                "First dimension of `rotation_weights` and `coupling_weights` must be the same"
            )

        if pattern in OPTIONS:
            expected_shape = (repeat_coupling, n_parameters[pattern])
            check_shape(
                coupling_weights,
                expected_shape,
                msg="'coupling_weights' must be of shape {}; got {}"
                "".format(expected_shape, get_shape(coupling_weights)),
            )

        elif isinstance(pattern, list):
            expected_shape = (repeat_coupling, len(pattern))
            check_shape(
                coupling_weights,
                expected_shape,
                msg="'coupling_weights' must be of shape {}; got {}"
                "".format(expected_shape, get_shape(coupling_weights)),
            )

    ###############

    for layer in range(repeat):

        broadcast(
            unitary=rotation, pattern="single", wires=wires, parameters=rotation_weights[layer]
        )
        if coupling_weights is not None:
            broadcast(
                unitary=coupling, pattern=pattern, wires=wires, parameters=coupling_weights[layer]
            )
        else:
            broadcast(unitary=coupling, pattern=pattern, wires=wires)
