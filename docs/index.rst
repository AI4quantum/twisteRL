.. TwisteRL documentation master file

TwisteRL Documentation
======================

.. image:: ../assets/twisterl-logo.png
   :width: 200
   :align: center
   :alt: TwisteRL

Welcome to TwisteRL
-------------------

TwisteRL is a minimalistic, high-performance Reinforcement Learning framework implemented in Rust with Python bindings.

The current version is a *Proof of Concept*, stay tuned for future releases!

🚀 Key Features
---------------

- **High-Performance Core**: RL episode loop implemented in Rust for faster training and inference
- **Inference-Ready**: Easy compilation and bundling of models with environments into portable binaries for inference
- **Modular Design**: Support for multiple algorithms (PPO, AlphaZero) with interchangeable training and inference
- **Language Interoperability**: Core in Rust with Python interface

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install twisterl

Training
~~~~~~~~

.. code-block:: bash

   python -m twisterl.train --config examples/ppo_puzzle8_v1.json

This example trains a model to play the popular "8 puzzle" where numbers have to be shifted around through the empty slot until they are in order.

This model can be trained on a single CPU in under 1 minute (no GPU required!).

🏗️ Current State (PoC)
-----------------------

- Hybrid rust-python implementation:
    - Data collection and inference in Rust
    - Training in Python (PyTorch)
- Supported algorithms:
    - PPO (Proximal Policy Optimization)
    - AlphaZero
- Focus on discrete observation and action spaces
- Support for native Rust environments and for Python environments through a wrapper

Getting Started
---------------

Ready to dive in? Here are the essential links to get you up and running:

📦 :doc:`installation` - Install TwisteRL and set up your environment

⚡ :doc:`quickstart` - Your first RL model in minutes

📖 :doc:`examples` - Interactive examples and use cases

🧠 :doc:`algorithms` - PPO and AlphaZero algorithm guides

Documentation
-------------

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   installation
   quickstart
   examples
   algorithms

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   api/twisterl
   api/environments
   api/algorithms
   api/neural_networks

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Development

   contributing
   roadmap
   changelog

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Additional Information

   docs-guide
   license

**API Reference**

- :doc:`api/twisterl` - Core package and functions
- :doc:`api/algorithms` - Algorithm implementations (PPO, AlphaZero)
- :doc:`api/environments` - Built-in and custom environments
- :doc:`api/neural_networks` - Neural network architectures

**Development & Community**

- :doc:`contributing` - How to contribute to TwisteRL
- :doc:`roadmap` - Development timeline and future features
- :doc:`changelog` - Version history and changes
- :doc:`docs-guide` - Building and deploying documentation

**Additional Resources**

- :ref:`genindex` - Complete index of functions and classes
- :ref:`search` - Search the documentation