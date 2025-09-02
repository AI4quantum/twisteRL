Installation Guide
==================

Prerequisites
-------------

TwisteRL requires:

- Python 3.9 or higher
- Rust toolchain (for building from source)
- PyTorch 2.2 or higher

From PyPI
-----------------------

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

3. Install the package:

.. code-block:: bash

   pip install .

Development Installation
------------------------

For development, install in editable mode with development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

Verify Installation
-------------------

Test your installation by running:

.. code-block:: python

   import twisterl
   print(twisterl.__version__)

Or try the quick example:

.. code-block:: bash

   python -m twisterl.train --config examples/ppo_puzzle8_v1.json