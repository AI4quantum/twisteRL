Algorithms API
==============

This section documents the internal algorithm implementations in TwisteRL.

RL Module
---------

.. automodule:: twisterl.rl
   :members:
   :undoc-members:
   :show-inheritance:

PPO Implementation
------------------

.. autoclass:: twisterl.rl.PPOAgent
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
~~~~~~~~~~~~~

The PPO algorithm accepts the following configuration parameters:

.. code-block:: python

   ppo_config = {
       "learning_rate": 0.0003,      # Learning rate for optimizer
       "clip_epsilon": 0.2,          # PPO clipping parameter  
       "value_loss_coef": 0.5,       # Value function loss coefficient
       "entropy_coef": 0.01,         # Entropy regularization coefficient
       "max_grad_norm": 0.5,         # Maximum gradient norm for clipping
       "num_epochs": 10,             # Training epochs per update
       "batch_size": 64,             # Mini-batch size
       "gamma": 0.99,                # Discount factor
       "gae_lambda": 0.95,           # GAE lambda parameter
       "normalize_advantages": True   # Whether to normalize advantages
   }

Methods
~~~~~~~

.. automethod:: twisterl.rl.PPOAgent.predict
.. automethod:: twisterl.rl.PPOAgent.train_step
.. automethod:: twisterl.rl.PPOAgent.save
.. automethod:: twisterl.rl.PPOAgent.load

AlphaZero Implementation
------------------------

.. autoclass:: twisterl.rl.AlphaZeroAgent
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   alphazero_config = {
       "num_simulations": 800,       # MCTS simulations per move
       "c_puct": 1.0,               # Exploration constant
       "temperature": 1.0,          # Action selection temperature
       "num_self_play_games": 1000, # Self-play games per iteration
       "training_steps": 1000,      # Training steps per iteration
       "batch_size": 32,            # Neural network batch size
       "learning_rate": 0.001,      # Neural network learning rate
       "value_loss_weight": 1.0,    # Value loss weight
       "regularization": 0.0001     # L2 regularization
   }

Monte Carlo Tree Search
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: twisterl.rl.MCTS
   :members:
   :undoc-members:
   :show-inheritance:

.. automethod:: twisterl.rl.MCTS.search
.. automethod:: twisterl.rl.MCTS.select_action
.. automethod:: twisterl.rl.MCTS.backup

Base Classes
------------

Agent Base Class
~~~~~~~~~~~~~~~~

.. autoclass:: twisterl.rl.BaseAgent
   :members:
   :undoc-members:
   :show-inheritance:

All algorithm implementations inherit from ``BaseAgent`` and must implement:

- ``predict(observation)`` - Get action for given observation
- ``train_step(batch)`` - Perform one training step  
- ``save(path)`` - Save agent to file
- ``load(path)`` - Load agent from file

Policy Networks
~~~~~~~~~~~~~~~

.. autoclass:: twisterl.rl.PolicyNetwork
   :members:
   :undoc-members:
   :show-inheritance:

Value Networks
~~~~~~~~~~~~~~

.. autoclass:: twisterl.rl.ValueNetwork
   :members:
   :undoc-members:
   :show-inheritance:

Training Loop
-------------

The core training loop is implemented in Rust for performance, but exposes a Python interface:

.. autofunction:: twisterl.rl.train_agent

Example usage:

.. code-block:: python

   import twisterl

   # Configure algorithm
   config = {
       "algorithm": "ppo",
       "environment": "puzzle8_v1", 
       "ppo": {
           "learning_rate": 0.0003,
           "num_epochs": 10
       },
       "training": {
           "total_timesteps": 100000,
           "eval_frequency": 10000,
           "save_frequency": 25000
       }
   }

   # Train agent
   agent = twisterl.train(config)

   # Or use the lower-level interface
   from twisterl.rl import train_agent, PPOAgent

   env = twisterl.make_env("puzzle8_v1")
   agent = PPOAgent(env.observation_space, env.action_space)
   trained_agent = train_agent(agent, env, total_timesteps=100000)

Data Collection
---------------

Episode data collection is handled by the Rust core for performance:

.. autoclass:: twisterl.rl.EpisodeCollector
   :members:
   :undoc-members:
   :show-inheritance:

.. automethod:: twisterl.rl.EpisodeCollector.collect_episodes
.. automethod:: twisterl.rl.EpisodeCollector.collect_rollouts

The collector returns structured data:

.. code-block:: python

   rollout_data = {
       "observations": numpy.ndarray,  # Shape: (timesteps, *obs_shape)  
       "actions": numpy.ndarray,       # Shape: (timesteps, *action_shape)
       "rewards": numpy.ndarray,       # Shape: (timesteps,)
       "dones": numpy.ndarray,         # Shape: (timesteps,)
       "values": numpy.ndarray,        # Shape: (timesteps,) [PPO only]
       "log_probs": numpy.ndarray,     # Shape: (timesteps,) [PPO only] 
       "advantages": numpy.ndarray,    # Shape: (timesteps,) [computed]
       "returns": numpy.ndarray        # Shape: (timesteps,) [computed]
   }

Metrics and Logging
-------------------

.. autoclass:: twisterl.rl.TrainingMetrics
   :members:
   :undoc-members:
   :show-inheritance:

Training metrics are automatically logged and can be viewed with TensorBoard:

.. code-block:: bash

   tensorboard --logdir runs/

Common metrics include:

- ``episode_reward_mean`` - Average episode reward
- ``episode_length_mean`` - Average episode length  
- ``policy_loss`` - Policy network loss
- ``value_loss`` - Value network loss [PPO]
- ``entropy`` - Policy entropy
- ``explained_variance`` - Value function explained variance [PPO]
- ``learning_rate`` - Current learning rate

Utilities
---------

.. autofunction:: twisterl.rl.compute_gae
.. autofunction:: twisterl.rl.compute_returns
.. autofunction:: twisterl.rl.normalize_advantages

Example of manual advantage computation:

.. code-block:: python

   from twisterl.rl import compute_gae, normalize_advantages

   # Compute GAE advantages
   advantages = compute_gae(
       rewards=rewards,
       values=values, 
       dones=dones,
       gamma=0.99,
       gae_lambda=0.95
   )

   # Normalize advantages  
   advantages = normalize_advantages(advantages)