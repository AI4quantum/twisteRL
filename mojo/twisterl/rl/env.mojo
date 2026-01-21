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


# In Mojo, traits define the interface that environments must implement.
# Since Mojo's trait system is evolving, we define the expected interface
# through documentation and provide helper types.


trait Env(Movable):
    """
    Trait defining the interface for reinforcement learning environments.

    All environments should implement these methods.
    """

    fn num_actions(self) -> Int:
        """Returns the number of possible actions."""
        ...

    fn obs_shape(self) -> List[Int]:
        """Returns the shape of observations."""
        ...

    fn set_difficulty(mut self, difficulty: Int):
        """Sets the current difficulty."""
        ...

    fn get_difficulty(self) -> Int:
        """Returns current difficulty."""
        ...

    fn set_state(mut self, state: List[Int]):
        """Sets the environment to a given state."""
        ...

    fn reset(mut self):
        """Resets the environment to a random initial state."""
        ...

    fn step(mut self, action: Int):
        """Evolves the current state by an action."""
        ...

    fn masks(self) -> List[Bool]:
        """Returns action masks (True if action is allowed)."""
        ...

    fn is_final(self) -> Bool:
        """Returns True if the current state is terminal."""
        ...

    fn reward(self) -> Float32:
        """Returns the reward for the current state."""
        ...

    fn observe(self) -> List[Int]:
        """Returns current state encoded in a sparse format."""
        ...

    fn clone(self) -> Self:
        """Creates a copy of this environment."""
        ...


# Helper functions for environments that don't implement the full trait
fn default_masks(num_actions: Int) -> List[Bool]:
    """Default implementation returning all actions as valid."""
    var masks = List[Bool]()
    for _ in range(num_actions):
        masks.append(True)
    return masks


fn default_twists() -> Tuple[List[List[Int]], List[List[Int]]]:
    """Default implementation returning empty permutation lists."""
    return (List[List[Int]](), List[List[Int]]())
