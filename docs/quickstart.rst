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

The training configuration is specified in JSON format. Here's an example based on the actual config structure:

.. code-block:: json

   {
       "env_cls": "twisterl.envs.Puzzle",
       "env": {
           "difficulty": 1,
           "height": 3,
           "width": 3,
           "depth_slope": 2,
           "max_depth": 256
       },
       "policy_cls": "twisterl.nn.BasicPolicy",
       "policy": {
           "embedding_size": 512,
           "common_layers": [256],
           "policy_layers": [],
           "value_layers": []
       },
       "algorithm_cls": "twisterl.rl.PPO",
       "algorithm": {
           "collecting": {
               "num_cores": 32,
               "num_episodes": 1024
           },
           "training": {
               "num_epochs": 10,
               "vf_coef": 0.8,
               "ent_coef": 0.01,
               "clip_ratio": 0.1,
               "normalize_advantage": true
           },
           "learning": {
               "diff_threshold": 0.85,
               "diff_max": 32
           },
           "optimizer": {
               "lr": 0.00015
           }
       }
   }

Training Options
~~~~~~~~~~~~~~~~

The training script accepts the following command-line arguments:

.. code-block:: bash

   python -m twisterl.train --config <path>           # Path to config file (required)
   python -m twisterl.train --config <path> --run_path <path>  # Custom output directory
   python -m twisterl.train --config <path> --load_checkpoint_path <path>  # Resume from checkpoint
   python -m twisterl.train --config <path> --num_steps <n>  # Limit training steps

Inference
~~~~~~~~~

After training, check the ``examples/puzzle.ipynb`` notebook for an interactive example showing how to:

- Load trained models
- Run inference
- Visualize agent behavior

Examples
--------

Check out the ``examples/`` directory for more comprehensive examples:

- **puzzle.ipynb**: Interactive Jupyter notebook showing inference
- **ppo_puzzle8_v1.json**: 8-puzzle training configuration
- **ppo_puzzle15_v1.json**: 15-puzzle training configuration (more challenging)
- **hub_puzzle_model.ipynb**: Loading models from HuggingFace Hub

Next Steps
----------

- Explore different :doc:`algorithms` (PPO, AlphaZero)
- Check out the full :doc:`api/twisterl` API reference
- Learn about :doc:`api/environments` for custom environments
