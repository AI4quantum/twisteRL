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

from collections import List, InlineArray
from random import random_ui64, random_si64


# Optimized version using InlineArray for fixed-size 3x3 puzzle
# This avoids heap allocations for the state array
alias PUZZLE_SIZE = 9  # 3x3 puzzle
alias NUM_ACTIONS = 4

# Pre-computed identity state for fast reset
alias _IDENTITY_STATE = InlineArray[Int, PUZZLE_SIZE](0, 1, 2, 3, 4, 5, 6, 7, 8)


struct PuzzleEnv(Copyable, Movable):
    """Sliding puzzle environment for reinforcement learning.

    Optimized implementation using InlineArray for stack allocation
    instead of List (heap allocation).
    """

    var state: InlineArray[Int, PUZZLE_SIZE]
    var zero_x: Int
    var zero_y: Int
    var depth: Int

    var width: Int
    var height: Int
    var difficulty: Int
    var depth_slope: Int
    var max_depth: Int

    fn __init__(
        out self,
        width: Int,
        height: Int,
        difficulty: Int,
        depth_slope: Int,
        max_depth: Int,
    ):
        self.width = width
        self.height = height
        self.difficulty = difficulty
        self.depth_slope = depth_slope
        self.max_depth = max_depth
        # Initialize state with identity permutation
        self.state = InlineArray[Int, PUZZLE_SIZE](fill=0)
        for i in range(PUZZLE_SIZE):
            self.state[i] = i
        self.zero_x = 0
        self.zero_y = 0
        self.depth = 1

    @always_inline
    fn solved(self) -> Bool:
        """Check if the puzzle is in solved state."""
        # Unrolled comparison for better performance
        for i in range(PUZZLE_SIZE):
            if self.state[i] != i:
                return False
        return True

    fn get_state(self) -> List[Int]:
        """Return a copy of the current state as List for compatibility."""
        var result = List[Int](capacity=PUZZLE_SIZE)
        for i in range(PUZZLE_SIZE):
            result.append(self.state[i])
        return result

    fn display(self):
        """Display the puzzle in a formatted way."""
        for i in range(PUZZLE_SIZE):
            var v = self.state[i]
            if v == 0:
                print("   ", end="")
            elif v < 10:
                print("  ", v, " ", end="")
            else:
                print(" ", v, " ", end="")
            if (i + 1) % self.width == 0:
                print("")

    @always_inline
    fn set_position(mut self, x: Int, y: Int, val: Int):
        """Set the value at position (x, y)."""
        self.state[y * self.width + x] = val

    @always_inline
    fn get_position(self, x: Int, y: Int) -> Int:
        """Get the value at position (x, y)."""
        return self.state[y * self.width + x]

    @always_inline
    fn num_actions(self) -> Int:
        """Return the number of possible actions (4 directions)."""
        return NUM_ACTIONS

    fn obs_shape(self) -> List[Int]:
        """Return the observation shape."""
        var shape = List[Int](capacity=2)
        shape.append(PUZZLE_SIZE)
        shape.append(PUZZLE_SIZE)
        return shape

    fn set_difficulty(mut self, difficulty: Int):
        """Set the difficulty level."""
        self.difficulty = difficulty

    @always_inline
    fn get_difficulty(self) -> Int:
        """Get the current difficulty level."""
        return self.difficulty

    fn set_state(mut self, state: List[Int]):
        """Set the puzzle state from a list."""
        for i in range(min(len(state), PUZZLE_SIZE)):
            self.state[i] = state[i]
        self.depth = self.max_depth

        for i in range(PUZZLE_SIZE):
            if self.state[i] == 0:
                self.zero_x = i % self.width
                self.zero_y = i // self.width
                break

    @always_inline
    fn reset(mut self):
        """Reset to initial state and apply random actions based on difficulty.

        Optimized: Uses pre-computed identity state and batched random generation.
        """
        # Fast copy from pre-computed identity state
        self.state = _IDENTITY_STATE
        self.zero_x = 0
        self.zero_y = 0

        # Apply random actions based on difficulty
        # Optimization: Generate one random number and extract bits for multiple actions
        if self.difficulty > 0:
            # Get a single random value and use different bits for each action
            # This is faster than calling random_ui64 multiple times
            var rand_bits = random_ui64(0, UInt64.MAX)
            for i in range(self.difficulty):
                # Extract 2 bits (0-3) for each action using bit shifting
                var action = Int((rand_bits >> (i * 2)) & 3)
                self._step_unchecked(action)

        self.depth = self.depth_slope * self.difficulty

    @always_inline
    fn _step_unchecked(mut self, action: Int):
        """Execute an action without bounds checking (internal use only)."""
        var zx = self.zero_x
        var zy = self.zero_y

        if action == 0 and zx > 0:
            var idx_curr = zy * self.width + zx
            var idx_new = zy * self.width + (zx - 1)
            self.state[idx_curr] = self.state[idx_new]
            self.state[idx_new] = 0
            self.zero_x = zx - 1
        elif action == 1 and zy > 0:
            var idx_curr = zy * self.width + zx
            var idx_new = (zy - 1) * self.width + zx
            self.state[idx_curr] = self.state[idx_new]
            self.state[idx_new] = 0
            self.zero_y = zy - 1
        elif action == 2 and zx < self.width - 1:
            var idx_curr = zy * self.width + zx
            var idx_new = zy * self.width + (zx + 1)
            self.state[idx_curr] = self.state[idx_new]
            self.state[idx_new] = 0
            self.zero_x = zx + 1
        elif action == 3 and zy < self.height - 1:
            var idx_curr = zy * self.width + zx
            var idx_new = (zy + 1) * self.width + zx
            self.state[idx_curr] = self.state[idx_new]
            self.state[idx_new] = 0
            self.zero_y = zy + 1

    @always_inline
    fn step(mut self, action: Int):
        """Execute an action (0=left, 1=up, 2=right, 3=down)."""
        var zx = self.zero_x
        var zy = self.zero_y

        if action == 0 and zx > 0:
            var new_val = self.get_position(zx - 1, zy)
            self.set_position(zx, zy, new_val)
            self.set_position(zx - 1, zy, 0)
            self.zero_x = zx - 1
        elif action == 1 and zy > 0:
            var new_val = self.get_position(zx, zy - 1)
            self.set_position(zx, zy, new_val)
            self.set_position(zx, zy - 1, 0)
            self.zero_y = zy - 1
        elif action == 2 and zx < self.width - 1:
            var new_val = self.get_position(zx + 1, zy)
            self.set_position(zx, zy, new_val)
            self.set_position(zx + 1, zy, 0)
            self.zero_x = zx + 1
        elif action == 3 and zy < self.height - 1:
            var new_val = self.get_position(zx, zy + 1)
            self.set_position(zx, zy, new_val)
            self.set_position(zx, zy + 1, 0)
            self.zero_y = zy + 1

        if self.depth > 0:
            self.depth -= 1

    @always_inline
    fn masks(self) -> InlineArray[Bool, NUM_ACTIONS]:
        """Return action masks (True if action is valid).

        Uses InlineArray for stack allocation instead of List.
        """
        var m = InlineArray[Bool, NUM_ACTIONS](fill=False)
        m[0] = self.zero_x > 0  # left
        m[1] = self.zero_y > 0  # up
        m[2] = self.zero_x < self.width - 1  # right
        m[3] = self.zero_y < self.height - 1  # down
        return m

    fn masks_list(self) -> List[Bool]:
        """Return action masks as List for compatibility."""
        var m = List[Bool](capacity=NUM_ACTIONS)
        m.append(self.zero_x > 0)  # left
        m.append(self.zero_y > 0)  # up
        m.append(self.zero_x < self.width - 1)  # right
        m.append(self.zero_y < self.height - 1)  # down
        return m

    @always_inline
    fn is_final(self) -> Bool:
        """Check if the episode is complete."""
        return self.depth == 0 or self.solved()

    @always_inline
    fn reward(self) -> Float32:
        """Return the reward for the current state."""
        if self.solved():
            return 1.0
        else:
            if self.depth == 0:
                return -0.5
            else:
                return -0.5 / Float32(self.max_depth)

    @always_inline
    fn observe(self) -> InlineArray[Int, PUZZLE_SIZE]:
        """Return the observation encoding using InlineArray.

        Optimized to avoid heap allocation.
        """
        var obs = InlineArray[Int, PUZZLE_SIZE](fill=0)
        var size = self.height * self.width
        for i in range(PUZZLE_SIZE):
            obs[i] = i * size + self.state[i]
        return obs

    fn observe_list(self) -> List[Int]:
        """Return the observation encoding as List for compatibility."""
        var obs = List[Int](capacity=PUZZLE_SIZE)
        var size = self.height * self.width
        for i in range(PUZZLE_SIZE):
            obs.append(i * size + self.state[i])
        return obs

    @always_inline
    fn clone(self) -> Self:
        """Create a copy of this environment.

        Optimized: InlineArray copy is much faster than List copy.
        """
        var new_env = PuzzleEnv(
            self.width,
            self.height,
            self.difficulty,
            self.depth_slope,
            self.max_depth,
        )
        # InlineArray copy is fast (stack copy)
        new_env.state = self.state
        new_env.zero_x = self.zero_x
        new_env.zero_y = self.zero_y
        new_env.depth = self.depth
        return new_env


# Alias for backward compatibility
alias Puzzle = PuzzleEnv
