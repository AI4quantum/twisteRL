Algorithms
==========

TwisteRL currently supports two reinforcement learning algorithms.

PPO (Proximal Policy Optimization)
-----------------------------------

PPO is a policy gradient method that strikes a balance between ease of implementation, sample complexity, and wall-clock time.

Key Features
~~~~~~~~~~~~

- **Stable Training**: Uses a clipped objective to prevent destructively large policy updates
- **Sample Efficient**: Reuses experience data multiple times through multiple epochs
- **General Purpose**: Works well across a wide variety of environments

Configuration
~~~~~~~~~~~~~

PPO is configured through a JSON config file. Here's an example with the actual parameter names:

.. code-block:: json

   {
       "algorithm_cls": "twisterl.rl.PPO",
       "algorithm": {
           "collecting": {
               "num_cores": 32,
               "num_episodes": 1024,
               "lambda": 0.995,
               "gamma": 0.995
           },
           "training": {
               "num_epochs": 10,
               "vf_coef": 0.8,
               "ent_coef": 0.01,
               "clip_ratio": 0.1,
               "normalize_advantage": true
           },
           "optimizer": {
               "lr": 0.00015
           },
           "learning": {
               "diff_threshold": 0.85,
               "diff_max": 32,
               "diff_metric": "ppo_1"
           }
       }
   }

Parameters
~~~~~~~~~~

**Collecting Parameters:**

- **num_cores**: Number of parallel workers for data collection
- **num_episodes**: Number of episodes to collect per iteration
- **lambda**: GAE lambda parameter for advantage estimation
- **gamma**: Discount factor

**Training Parameters:**

- **num_epochs**: Number of training epochs per update
- **vf_coef**: Coefficient for value function loss
- **ent_coef**: Coefficient for entropy bonus
- **clip_ratio**: PPO clipping parameter (epsilon)
- **normalize_advantage**: Whether to normalize advantages

**Optimizer Parameters:**

- **lr**: Learning rate for Adam optimizer

**Learning Parameters:**

- **diff_threshold**: Success rate threshold for increasing difficulty
- **diff_max**: Maximum difficulty level
- **diff_metric**: Which evaluation metric to use for difficulty progression

Example
~~~~~~~

Train PPO on the 8-puzzle:

.. code-block:: bash

   python -m twisterl.train --config examples/ppo_puzzle8_v1.json

AlphaZero (AZ)
--------------

AlphaZero combines Monte Carlo Tree Search (MCTS) with deep neural networks for planning-based learning.

Key Features
~~~~~~~~~~~~

- **Tree Search**: Employs MCTS for look-ahead planning
- **Self-Play**: Learns through self-play without human knowledge
- **Value + Policy Learning**: Jointly learns value and policy functions

Configuration
~~~~~~~~~~~~~

AlphaZero is configured similarly to PPO:

.. code-block:: json

   {
       "algorithm_cls": "twisterl.rl.AZ",
       "algorithm": {
           "collecting": {
               "num_cores": 32,
               "num_episodes": 512,
               "num_mcts_searches": 1000,
               "C": 1.41,
               "max_expand_depth": 1,
               "seed": 123
           },
           "training": {
               "num_epochs": 10
           },
           "optimizer": {
               "lr": 0.0003
           }
       }
   }

Parameters
~~~~~~~~~~

**Collecting Parameters:**

- **num_mcts_searches**: Number of MCTS simulations per move
- **C**: Exploration constant (UCB formula)
- **max_expand_depth**: Maximum tree expansion depth
- **seed**: Random seed for reproducibility

Algorithm Comparison
--------------------

+----------------+----------+---------------+----------------+-----------------+
| Algorithm      | Type     | Sample Eff.   | Compute Cost   | Use Case        |
+================+==========+===============+================+=================+
| PPO            | On-Policy| Medium        | Low            | General RL      |
+----------------+----------+---------------+----------------+-----------------+
| AlphaZero      | Planning | High          | High           | Perfect Info    |
+----------------+----------+---------------+----------------+-----------------+

When to Use Each Algorithm
--------------------------

**Use PPO when:**

- You want a general-purpose, fast-to-train algorithm
- Computational resources are limited
- You need stable, reliable training

**Use AlphaZero when:**

- The environment has perfect information (you know the transition model)
- You can afford higher computational cost for MCTS
- Look-ahead planning is beneficial

Hyperparameter Tuning
----------------------

General Guidelines
~~~~~~~~~~~~~~~~~~

1. **Start with defaults**: See ``src/twisterl/defaults.py`` for sensible default parameters
2. **Adjust learning rate first**: This usually has the biggest impact
3. **Monitor training curves**: Use TensorBoard to track progress (logs saved to ``runs/`` by default)

PPO Tuning Tips
~~~~~~~~~~~~~~~

- Increase ``num_epochs`` if training is stable but slow
- Decrease ``clip_ratio`` if policy updates are too aggressive
- Increase ``ent_coef`` if the policy becomes too deterministic too quickly
- Adjust ``gamma`` and ``lambda`` based on episode length

AlphaZero Tuning Tips
~~~~~~~~~~~~~~~~~~~~~

- Increase ``num_mcts_searches`` for better play quality (but slower training)
- Adjust ``C`` to balance exploration vs exploitation in MCTS
