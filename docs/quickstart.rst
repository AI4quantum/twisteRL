Quick Start Guide
=================

Basic Usage
-----------

Training Your First Model
~~~~~~~~~~~~~~~~~~~~~~~~~

TwisteRL comes with built-in puzzle environments that are perfect for getting started:

.. code-block:: bash

   python -m twisterl.train --config examples/ppo_puzzle8_v1.json

This will train a PPO agent to solve the classic 8-puzzle:

.. code-block:: text

   |8|7|5|
   |3|2| |
   |4|6|1|

The goal is to rearrange the numbers by sliding them into the empty space until they're in numerical order.

Training Configuration
~~~~~~~~~~~~~~~~~~~~~~

The training configuration is specified in JSON format. Here's a basic example:

.. code-block:: json

   {
     "algorithm": "ppo",
     "environment": "puzzle8_v1",
     "training": {
       "total_timesteps": 100000,
       "learning_rate": 0.0003,
       "batch_size": 64
     },
     "network": {
       "hidden_layers": [128, 128],
       "activation": "relu"
     }
   }

Running Inference
~~~~~~~~~~~~~~~~~

After training, you can test your trained model:

.. code-block:: python

   import twisterl
   
   # Load trained model
   agent = twisterl.load_agent("path/to/model")
   
   # Create environment
   env = twisterl.make_env("puzzle8_v1")
   
   # Run episode
   obs = env.reset()
   done = False
   total_reward = 0
   
   while not done:
       action = agent.predict(obs)
       obs, reward, done, info = env.step(action)
       total_reward += reward
   
   print(f"Total reward: {total_reward}")

Examples
--------

Check out the `examples/` directory for more comprehensive examples:

- **puzzle.ipynb**: Interactive Jupyter notebook showing inference
- **ppo_puzzle8_v1.json**: 8-puzzle training configuration  
- **ppo_puzzle15_v1.json**: 15-puzzle training configuration (more challenging)

Next Steps
----------

- Explore different :doc:`algorithms` (PPO, AlphaZero)
- Learn about custom :doc:`api/environments`
- Check out the full :doc:`api/twisterl` API reference