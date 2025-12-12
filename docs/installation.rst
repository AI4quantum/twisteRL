Installation Guide
==================

Prerequisites
-------------

TwisteRL requires:

- Python 3.9 or higher
- Rust toolchain (for building from source)
- PyTorch 2.2 or higher

From PyPI
---------

.. code-block:: bash

   pip install twisterl


From Source
-----------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/AI4quantum/twisteRL.git
   cd twisteRL

2. Create and activate a virtual environment:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install the Rust toolchain (if not already installed):

.. code-block:: bash

   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env

4. Install the package:

.. code-block:: bash

   pip install .

Development Installation
------------------------

For development, install in editable mode:

.. code-block:: bash

   pip install -e .

Verify Installation
-------------------

Test your installation by running the training example:

.. code-block:: bash

   python -m twisterl.train --config examples/ppo_puzzle8_v1.json

Or import the package in Python:

.. code-block:: python

   import twisterl
   from twisterl.rl import PPO, AZ
   from twisterl.nn import BasicPolicy
