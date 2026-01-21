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


fn relu(x: Float32) -> Float32:
    """ReLU activation function."""
    if x > 0.0:
        return x
    else:
        return 0.0


struct Linear(Copyable, Movable):
    """Linear (fully connected) layer."""
    var weights: List[Float32]
    var bias: List[Float32]
    var apply_relu: Bool
    var n_in: Int
    var n_out: Int

    fn __init__(
        out self,
        weights_vector: List[Float32],
        bias_vector: List[Float32],
        apply_relu: Bool,
    ):
        self.bias = List[Float32]()
        for i in range(len(bias_vector)):
            self.bias.append(bias_vector[i])

        self.weights = List[Float32]()
        for i in range(len(weights_vector)):
            self.weights.append(weights_vector[i])

        self.n_out = len(bias_vector)
        self.n_in = len(weights_vector) // self.n_out if self.n_out > 0 else 0
        self.apply_relu = apply_relu

    fn forward(self, input: List[Float32]) -> List[Float32]:
        """Forward pass through the linear layer."""
        var out = List[Float32]()

        for i in range(self.n_out):
            var sum_val = self.bias[i]
            for j in range(self.n_in):
                if j < len(input):
                    sum_val += self.weights[i * self.n_in + j] * input[j]

            if self.apply_relu and sum_val < 0:
                sum_val = 0.0
            out.append(sum_val)

        return out


struct EmbeddingBag(Copyable, Movable):
    """Embedding bag layer for sparse inputs."""
    var vectors: List[List[Float32]]
    var bias: List[Float32]
    var apply_relu: Bool
    var obs_shape: List[Int]
    var conv_dim: Int

    fn __init__(
        out self,
        vec_vectors: List[List[Float32]],
        bias_vector: List[Float32],
        apply_relu: Bool,
        obs_shape: List[Int],
        conv_dim: Int,
    ):
        self.vectors = List[List[Float32]]()
        for i in range(len(vec_vectors)):
            var vec = List[Float32]()
            for j in range(len(vec_vectors[i])):
                vec.append(vec_vectors[i][j])
            self.vectors.append(vec)

        self.bias = List[Float32]()
        for i in range(len(bias_vector)):
            self.bias.append(bias_vector[i])

        self.apply_relu = apply_relu

        self.obs_shape = List[Int]()
        for i in range(len(obs_shape)):
            self.obs_shape.append(obs_shape[i])

        self.conv_dim = conv_dim

    fn forward(self, input: List[Int]) -> List[Float32]:
        """Forward pass through the embedding bag."""
        var out = List[Float32]()
        for i in range(len(self.bias)):
            out.append(self.bias[i])

        if len(self.obs_shape) == 1:
            # 1D observation
            for idx in input:
                if idx < len(self.vectors):
                    var vec = self.vectors[idx]
                    for j in range(len(vec)):
                        if j < len(out):
                            out[j] += vec[j]

        elif len(self.obs_shape) == 2:
            # 2D observation
            var v_size = len(self.vectors[0]) if len(self.vectors) > 0 else 0

            for idx in input:
                var row = idx // self.obs_shape[1]
                var col = idx % self.obs_shape[1]

                if self.conv_dim == 1:
                    var temp = row
                    row = col
                    col = temp

                if row < len(self.vectors):
                    var vec = self.vectors[row]
                    for j in range(len(vec)):
                        var out_idx = col * v_size + j
                        if out_idx < len(out):
                            out[out_idx] += vec[j]

        if self.apply_relu:
            for i in range(len(out)):
                if out[i] < 0:
                    out[i] = 0.0

        return out
