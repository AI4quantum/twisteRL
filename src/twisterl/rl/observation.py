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

"""Thin Python wrapper around the Rust observation codec.

We keep a small shim in Python even though the heavy lifting happens in Rust.
This wrapper is the single place that:

* Converts collectors' nested lists / NumPy arrays into the plain lists that the
  PyO3 bindings expect, and turns the Rust output back into NumPy (most callers
  still work with tensors/arrays).
* Preserves the historical `self.obs_encoder(obs)` call pattern used by
  algorithms and tests, so we can swap codecs without touching every learner.
* Carries per-run dtype choices that may differ from the float32 default the
  Rust side returns (useful when training on CPU/GPU with different precision).

Keeping these concerns in one spot avoids scattering conversion boilerplate
around PPO/AZ and keeps configuration handling readable.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from twisterl import twisterl


class ObservationEncoder:
    """Wraps the Rust codec while hiding Python-facing glue (dtype/NumPy)."""

    def __init__(self, rust_codec, dtype: type = float):
        self._codec = rust_codec
        self._dtype = dtype

    def __call__(self, obs: Sequence[Iterable[int]] | np.ndarray) -> np.ndarray:
        if isinstance(obs, np.ndarray):
            if obs.size == 0:
                return self._empty_batch()
            obs_list = obs.tolist()
        else:
            obs_seq = list(obs)
            if not obs_seq:
                return self._empty_batch()
            obs_list = [list(sample) for sample in obs_seq]

        encoded = self._codec.encode(obs_list)
        if not encoded:
            return self._empty_batch()
        return np.asarray(encoded, dtype=self._dtype)

    def to_rust(self):
        return self._codec.clone()

    def _empty_batch(self) -> np.ndarray:
        obs_size_attr = getattr(self._codec, "obs_size")
        obs_size = obs_size_attr() if callable(obs_size_attr) else obs_size_attr
        return np.zeros((0, int(obs_size)), dtype=self._dtype)


def make_observation_encoder(
    obs_shape: Sequence[int], config=None
) -> ObservationEncoder:
    if config is None:
        raise ValueError("Observation encoder configuration must be provided.")

    if isinstance(config, str):
        encoder_type = config
        params = {}
    else:
        encoder_type = config.get("type")
        if encoder_type is None:
            raise ValueError("Observation encoder configuration must include a 'type'.")
        params = {k: v for k, v in config.items() if k != "type"}

    if encoder_type == "multi_hot":
        if len(obs_shape) < 2:
            raise ValueError(
                "Multi-hot encoder requires obs_shape with at least two elements."
            )
        num_slots = int(obs_shape[0])
        domain_size = int(obs_shape[1])
        dtype = params.get("dtype", float)
        rust_codec = twisterl.codec.make_observation_codec(
            "multi_hot", num_slots, domain_size
        )
        return ObservationEncoder(rust_codec, dtype=dtype)

    raise ValueError(f"Unknown observation encoder type: {encoder_type}")
