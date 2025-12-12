Environments
============

TwisteRL provides a flexible environment system supporting both native Rust environments and Python environment wrappers.

envs Module
-----------

.. automodule:: twisterl.envs
   :members:
   :undoc-members:
   :show-inheritance:

Built-in Environments
----------------------

Puzzle Environment
~~~~~~~~~~~~~~~~~~

The ``Puzzle`` environment is a sliding tile puzzle implemented in Rust.

**Configuration:**

.. code-block:: json

   {
       "env_cls": "twisterl.envs.Puzzle",
       "env": {
           "difficulty": 1,
           "height": 3,
           "width": 3,
           "depth_slope": 2,
           "max_depth": 256
       }
   }

**Parameters:**

- **difficulty**: Initial difficulty level (controls scramble depth)
- **height**: Grid height (3 for 8-puzzle, 4 for 15-puzzle)
- **width**: Grid width (3 for 8-puzzle, 4 for 15-puzzle)
- **depth_slope**: How quickly difficulty increases scramble depth
- **max_depth**: Maximum scramble depth

**8-Puzzle (3x3):**

A 3x3 sliding puzzle with 8 numbered tiles and one empty space.

**15-Puzzle (4x4):**

A 4x4 sliding puzzle with 15 numbered tiles and one empty space.

Python Environment Wrapper
--------------------------

The ``PyEnv`` class wraps Python environments for use with TwisteRL's Rust training loop.

**Configuration:**

.. code-block:: json

   {
       "env_cls": "twisterl.envs.PyEnv",
       "env": {
           "pyenv_cls": "mymodule.MyEnvironment"
       }
   }

The Python environment class must implement:

- ``reset(difficulty: int)``: Reset the environment to initial state with given difficulty
- ``next(action: int)``: Execute an action (advances the environment state)
- ``observe() -> list[int]``: Return the current observation
- ``obs_shape() -> list[int]``: Return observation dimensions
- ``num_actions() -> int``: Return number of valid actions
- ``is_final() -> bool``: Return True if current state is terminal
- ``success() -> bool``: Return True if the goal was achieved
- ``value() -> float``: Return the reward value for current state
- ``masks() -> list[bool]``: Return action mask (True if action is valid)
- ``set_state(state: list[int])``: Set environment to specific state
- ``copy()``: Return a copy of the environment (for parallel collection)
- ``twists() -> (obs_perms, act_perms)`` (optional, for symmetry-aware training)

Creating Custom Environments
----------------------------

For best performance, implement environments in Rust. See the ``examples/grid_world`` directory for a complete example.

**Key steps:**

1. Implement the ``twisterl::rl::env::Env`` trait in Rust
2. Expose to Python using ``PyBaseEnv``
3. Build with maturin and install

See :doc:`../examples` for detailed instructions.

Environment Interface (Rust Trait)
-----------------------------------

Rust environments implement the ``twisterl::rl::env::Env`` trait. The required methods are:

- ``num_actions() -> usize``: Return number of possible actions
- ``obs_shape() -> Vec<usize>``: Return observation dimensions
- ``set_state(state: Vec<i64>)``: Set environment to a specific state
- ``reset()``: Reset to a random initial state
- ``step(action: usize)``: Execute an action (evolve the state)
- ``is_final() -> bool``: Return True if current state is terminal
- ``success() -> bool``: Return True if the goal was achieved
- ``reward() -> f32``: Return the reward value for current state
- ``observe() -> Vec<usize>``: Return current state as sparse observation

Optional methods with default implementations:

- ``set_difficulty(difficulty: usize)``: Set difficulty level (default: no-op)
- ``get_difficulty() -> usize``: Get current difficulty (default: 1)
- ``masks() -> Vec<bool>``: Return action mask (default: all True)
- ``twists() -> (Vec<Vec<usize>>, Vec<Vec<usize>>)``: Return permutation symmetries (default: empty)

Permutation Symmetries (Twists)
-------------------------------

TwisteRL supports symmetry-aware training through "twists" - permutations of observations and actions that represent equivalent states.

See `twists.md <https://github.com/AI4quantum/twisteRL/blob/main/docs/twists.md>`_ for detailed documentation on implementing twists in your environments.
