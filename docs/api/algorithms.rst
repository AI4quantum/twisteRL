Algorithms API
==============

This section documents the algorithm implementations in TwisteRL.

RL Module
---------

.. automodule:: twisterl.rl
   :members:
   :undoc-members:
   :show-inheritance:

Algorithm Base Class
--------------------

.. automodule:: twisterl.rl.algorithm
   :members:
   :undoc-members:
   :show-inheritance:

The ``Algorithm`` base class provides the core training loop. Key methods:

- ``learn(num_steps, best_metrics=None)``: Main training loop
- ``learn_step()``: Single training iteration (collect, transform, train)
- ``collect()``: Collect rollout data using the Rust collector
- ``train(torch_data)``: Train for ``num_epochs`` calling ``train_step``
- ``evaluate(kwargs)``: Evaluate the current policy
- ``solve(state, deterministic, num_searches, num_mcts_searches, C, max_expand_depth)``: Solve from a given state

PPO Implementation
------------------

.. automodule:: twisterl.rl.ppo
   :members:
   :undoc-members:
   :show-inheritance:

The ``PPO`` class implements Proximal Policy Optimization.

**Key methods:**

- ``data_to_torch(data)``: Convert collected data to PyTorch tensors
- ``train_step(torch_data)``: Perform one gradient update

**Training losses:**

- Policy loss (clipped surrogate objective)
- Value function loss (MSE)
- Entropy bonus

AlphaZero Implementation
------------------------

.. automodule:: twisterl.rl.az
   :members:
   :undoc-members:
   :show-inheritance:

The ``AZ`` class implements AlphaZero with MCTS.

**Key methods:**

- ``data_to_torch(data)``: Convert MCTS data to PyTorch tensors
- ``train_step(torch_data)``: Train policy and value heads

Data Collection
---------------

Data collection is handled by Rust collectors (``twisterl.collector.PPOCollector`` and ``twisterl.collector.AZCollector``).

The collectors return data objects with:

- ``obs``: Observations
- ``logits``: Policy logits
- ``values``: Value predictions
- ``rewards``: Rewards
- ``actions``: Actions taken
- ``additional_data``: Algorithm-specific data (returns, advantages for PPO; remaining_values for AZ)

Training Loop
-------------

Training is run via the command line:

.. code-block:: bash

   python -m twisterl.train --config path/to/config.json

Or programmatically:

.. code-block:: python

   from twisterl.utils import prepare_algorithm, load_config

   config = load_config("path/to/config.json")
   algorithm = prepare_algorithm(config, run_path="runs/my_run")
   algorithm.learn(num_steps=1000)

Metrics and Logging
-------------------

Training metrics are logged to TensorBoard:

.. code-block:: bash

   tensorboard --logdir runs/

Logged metrics include:

- ``Benchmark/difficulty``: Current difficulty level
- ``Benchmark/success``: Success rate on evaluation
- ``Benchmark/reward``: Average reward
- ``Losses/value``: Value function loss
- ``Losses/policy``: Policy loss
- ``Losses/entropy``: Entropy (PPO only)
- ``Times/*``: Timing breakdown for each step

Checkpointing
-------------

Checkpoints are saved automatically in safetensors format:

- ``checkpoint_last.safetensors``: Most recent checkpoint (frequency controlled by ``logging.checkpoint_freq``)
- ``checkpoint_best.safetensors``: Best performing checkpoint

Load a checkpoint:

.. code-block:: bash

   python -m twisterl.train --config config.json --load_checkpoint_path runs/my_run/checkpoint_best.safetensors
