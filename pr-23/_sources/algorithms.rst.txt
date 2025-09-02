Algorithms
==========

TwisteRL currently supports two main reinforcement learning algorithms, with more planned for future releases.

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

.. code-block:: json

   {
     "algorithm": "ppo",
     "ppo": {
       "learning_rate": 0.0003,
       "clip_epsilon": 0.2,
       "value_loss_coef": 0.5,
       "entropy_coef": 0.01,
       "max_grad_norm": 0.5,
       "num_epochs": 10,
       "batch_size": 64,
       "gamma": 0.99,
       "gae_lambda": 0.95
     }
   }

Parameters
~~~~~~~~~~

- **learning_rate**: Learning rate for the optimizer
- **clip_epsilon**: PPO clipping parameter
- **value_loss_coef**: Coefficient for value function loss  
- **entropy_coef**: Coefficient for entropy bonus
- **max_grad_norm**: Maximum gradient norm for clipping
- **num_epochs**: Number of training epochs per update
- **batch_size**: Mini-batch size for training
- **gamma**: Discount factor
- **gae_lambda**: Lambda parameter for Generalized Advantage Estimation

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   import twisterl

   config = {
       "algorithm": "ppo",
       "environment": "puzzle8_v1",
       "ppo": {
           "learning_rate": 0.0003,
           "clip_epsilon": 0.2,
           "num_epochs": 10
       },
       "training": {
           "total_timesteps": 100000
       }
   }

   agent = twisterl.train(config)

AlphaZero
---------

AlphaZero is a model-based algorithm that combines Monte Carlo Tree Search (MCTS) with deep neural networks.

Key Features
~~~~~~~~~~~~

- **Model-Based**: Uses a learned model of the environment
- **Tree Search**: Employs MCTS for planning
- **Self-Play**: Improves through self-play without human knowledge

Configuration
~~~~~~~~~~~~~

.. code-block:: json

   {
     "algorithm": "alphazero",
     "alphazero": {
       "num_simulations": 800,
       "c_puct": 1.0,
       "temperature": 1.0,
       "num_self_play_games": 1000,
       "training_steps": 1000,
       "batch_size": 32,
       "learning_rate": 0.001
     }
   }

Parameters
~~~~~~~~~~

- **num_simulations**: Number of MCTS simulations per move
- **c_puct**: Exploration constant for MCTS
- **temperature**: Temperature for action selection
- **num_self_play_games**: Number of self-play games per iteration
- **training_steps**: Number of training steps per iteration
- **batch_size**: Mini-batch size for neural network training
- **learning_rate**: Learning rate for the neural network

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   import twisterl

   config = {
       "algorithm": "alphazero", 
       "environment": "puzzle8_v1",
       "alphazero": {
           "num_simulations": 800,
           "num_self_play_games": 500
       },
       "training": {
           "total_timesteps": 50000
       }
   }

   agent = twisterl.train(config)

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
- You want a general-purpose algorithm
- Computational resources are limited
- The environment is partially observable
- You need stable, reliable training

**Use AlphaZero when:**
- The environment has perfect information
- You can afford higher computational cost
- Sample efficiency is critical
- The environment is deterministic or nearly so

Hyperparameter Tuning
----------------------

General Guidelines
~~~~~~~~~~~~~~~~~~

1. **Start with defaults**: Both algorithms come with sensible default parameters
2. **Adjust learning rate first**: This usually has the biggest impact
3. **Monitor training curves**: Use TensorBoard to track progress
4. **Validate on multiple seeds**: Run multiple random seeds to ensure robustness

PPO Tuning Tips
~~~~~~~~~~~~~~~

- Increase ``num_epochs`` if training is stable but slow
- Decrease ``clip_epsilon`` if policy updates are too aggressive
- Increase ``entropy_coef`` if the policy becomes too deterministic too quickly

AlphaZero Tuning Tips
~~~~~~~~~~~~~~~~~~~~~

- Increase ``num_simulations`` for better play quality (but slower training)
- Adjust ``c_puct`` to balance exploration vs exploitation in MCTS
- Tune ``temperature`` schedule for better action exploration

Future Algorithms
------------------

Planned for future releases:

- **A3C/A2C**: Asynchronous advantage actor-critic methods
- **SAC**: Soft Actor-Critic for continuous control
- **TD3**: Twin Delayed Deep Deterministic Policy Gradient
- **Rainbow DQN**: Improved deep Q-learning with multiple extensions