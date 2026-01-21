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

from collections import List, Dict


struct CollectedData(Copyable, Movable):
    """Container for collected rollout data."""
    var obs: List[List[Int]]
    """Observations at each timestep: List of feature Lists"""
    var logits: List[List[Float32]]
    """Logits (action probabilities) at each timestep"""
    var values: List[Float32]
    """Value estimates at each timestep"""
    var rewards: List[Float32]
    """Rewards received at each timestep"""
    var actions: List[Int]
    """Actions taken at each timestep"""
    var advs: List[Float32]
    """Advantages (for PPO)"""
    var rets: List[Float32]
    """Returns (for PPO)"""
    var remaining_values: List[Float32]
    """Remaining values (for AZ)"""

    fn __init__(
        out self,
        obs: List[List[Int]],
        logits: List[List[Float32]],
        values: List[Float32],
        rewards: List[Float32],
        actions: List[Int],
    ):
        self.obs = obs
        self.logits = logits
        self.values = values
        self.rewards = rewards
        self.actions = actions
        self.advs = List[Float32]()
        self.rets = List[Float32]()
        self.remaining_values = List[Float32]()

    fn __init__(out self):
        """Create empty CollectedData."""
        self.obs = List[List[Int]]()
        self.logits = List[List[Float32]]()
        self.values = List[Float32]()
        self.rewards = List[Float32]()
        self.actions = List[Int]()
        self.advs = List[Float32]()
        self.rets = List[Float32]()
        self.remaining_values = List[Float32]()

    fn merge(mut self, other: CollectedData):
        """Merge another CollectedData into this one by appending all lists."""
        # Append observations
        for i in range(len(other.obs)):
            self.obs.append(other.obs[i])

        # Append logits
        for i in range(len(other.logits)):
            self.logits.append(other.logits[i])

        # Append 1D lists
        for i in range(len(other.values)):
            self.values.append(other.values[i])

        for i in range(len(other.rewards)):
            self.rewards.append(other.rewards[i])

        for i in range(len(other.actions)):
            self.actions.append(other.actions[i])

        # Append additional data
        for i in range(len(other.advs)):
            self.advs.append(other.advs[i])

        for i in range(len(other.rets)):
            self.rets.append(other.rets[i])

        for i in range(len(other.remaining_values)):
            self.remaining_values.append(other.remaining_values[i])

    fn len(self) -> Int:
        """Return the number of timesteps in the collected data."""
        return len(self.obs)


fn merge(owned chunks: List[CollectedData]) -> CollectedData:
    """Merge many episodes into one."""
    if len(chunks) == 0:
        return CollectedData()

    var merged = chunks.pop()

    while len(chunks) > 0:
        var chunk = chunks.pop()
        merged.merge(chunk)

    return merged
