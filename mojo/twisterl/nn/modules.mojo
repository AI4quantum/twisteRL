# -*- coding: utf-8 -*-
# (C) Copyright 2025 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from collections import List
from .layers import Linear


struct Sequential(Copyable, Movable):
    """Sequential container for neural network layers."""
    var weights: List[List[Float32]]
    var biases: List[List[Float32]]
    var apply_relus: List[Bool]

    fn __init__(out self):
        """Create an empty Sequential container."""
        self.weights = List[List[Float32]]()
        self.biases = List[List[Float32]]()
        self.apply_relus = List[Bool]()

    fn __init__(
        out self,
        weights: List[List[Float32]],
        biases: List[List[Float32]],
        apply_relus: List[Bool],
    ):
        """Create Sequential with pre-defined layers."""
        self.weights = weights
        self.biases = biases
        self.apply_relus = apply_relus

    fn add_layer(mut self, layer: Linear):
        """Add a linear layer to the sequential."""
        var w = List[Float32]()
        for i in range(len(layer.weights)):
            w.append(layer.weights[i])
        self.weights.append(w)

        var b = List[Float32]()
        for i in range(len(layer.bias)):
            b.append(layer.bias[i])
        self.biases.append(b)

        self.apply_relus.append(layer.apply_relu)

    fn forward(self, input: List[Float32]) -> List[Float32]:
        """Forward pass through all layers."""
        var x = input

        for layer_idx in range(len(self.weights)):
            var weights = self.weights[layer_idx]
            var bias = self.biases[layer_idx]
            var apply_relu = self.apply_relus[layer_idx]

            var n_out = len(bias)
            var n_in = len(weights) // n_out if n_out > 0 else 0

            var out = List[Float32]()
            for i in range(n_out):
                var sum_val = bias[i]
                for j in range(n_in):
                    if j < len(x):
                        sum_val += weights[i * n_in + j] * x[j]

                if apply_relu and sum_val < 0:
                    sum_val = 0.0
                out.append(sum_val)

            x = out

        return x

    fn num_layers(self) -> Int:
        """Return the number of layers."""
        return len(self.weights)
