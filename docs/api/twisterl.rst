twisterl package
================

.. automodule:: twisterl
   :members:
   :undoc-members:
   :show-inheritance:

Submodules
----------

.. toctree::
   :maxdepth: 4

   twisterl.rl
   twisterl.envs
   twisterl.nn

twisterl.train module
---------------------

The main training entry point. Run training via:

.. code-block:: bash

   python -m twisterl.train --config <path_to_config.json>

.. automodule:: twisterl.train
   :members:
   :undoc-members:
   :show-inheritance:

twisterl.utils module
---------------------

Utility functions for loading configs and preparing algorithms.

.. automodule:: twisterl.utils
   :members:
   :undoc-members:
   :show-inheritance:

Key functions:

- ``prepare_algorithm(config, run_path, load_checkpoint_path)``: Prepares an algorithm instance from config
- ``load_config(config_path)``: Loads a JSON config file
- ``pull_hub_algorithm(repo_id, model_path, revision, validate)``: Downloads models from HuggingFace Hub

twisterl.defaults module
------------------------

Default configuration values for algorithms, training, and evaluation.

.. automodule:: twisterl.defaults
   :members:
   :undoc-members:
   :show-inheritance:
