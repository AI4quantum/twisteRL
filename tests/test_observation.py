# -*- coding: utf-8 -*-

"""Unit tests for the Rust-backed observation encoder."""

import numpy as np

from twisterl.rl.observation import ObservationEncoder, make_observation_encoder


def test_make_observation_encoder_multi_hot():
    enc = make_observation_encoder([3, 2], {"type": "multi_hot", "dtype": float})
    assert isinstance(enc, ObservationEncoder)
    obs = [[0, 1], [4]]
    encoded = enc(obs)
    assert encoded.shape == (2, 6)
    assert np.all(encoded[0, [0, 1]] == 1.0)
    assert encoded[1, 4] == 1.0
    assert encoded.dtype == float


def test_make_observation_encoder_requires_type():
    try:
        make_observation_encoder([5, 5], {})
    except ValueError as err:
        assert "type" in str(err)
    else:
        raise AssertionError("Expected ValueError when encoder type is missing.")


def test_make_observation_encoder_unknown_type():
    try:
        make_observation_encoder([5, 5], {"type": "unknown"})
    except ValueError as err:
        assert "Unknown observation encoder type" in str(err)
    else:
        raise AssertionError("Expected ValueError for unknown encoder type.")


def test_observation_encoder_handles_empty_batch():
    enc = make_observation_encoder([3, 2], {"type": "multi_hot", "dtype": float})
    encoded = enc([])
    assert encoded.shape == (0, 6)
    assert encoded.dtype == float
