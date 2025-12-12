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

The Python environment class should implement:

- ``reset() -> observation``
- ``step(action) -> (observation, reward, done, info)``
- ``obs_shape() -> list[int]``
- ``num_actions() -> int``
- ``twists() -> (obs_perms, act_perms)`` (optional, for symmetry-aware training)

Creating Custom Environments
----------------------------

For best performance, implement environments in Rust. See the ``examples/grid_world`` directory for a complete example.

**Key steps:**

1. Implement the ``twisterl::rl::env::Env`` trait in Rust
2. Expose to Python using ``PyBaseEnv``
3. Build with maturin and install

See :doc:`../examples` for detailed instructions.

Environment Interface
---------------------

All environments must provide these methods (called from Rust or Python):

- ``reset()``: Reset to initial state, return observation
- ``step(action)``: Take action, return (obs, reward, done, info)
- ``obs_shape()``: Return observation dimensions
- ``num_actions()``: Return number of valid actions
- ``is_final()``: Return True if current state is terminal
- ``success()``: Return True if the goal was achieved (episode ended successfully)
- ``reward()``: Return the reward value for the current state
- ``twists()``: Return permutation symmetries (optional)
- ``set_state(state)``: Set environment to specific state (for inference)
- ``difficulty``: Property to get/set difficulty level

Permutation Symmetries (Twists)
-------------------------------

TwisteRL supports symmetry-aware training through "twists" - permutations of observations and actions that represent equivalent states.

See ``docs/twists.md`` for detailed documentation on implementing twists in your environments.
