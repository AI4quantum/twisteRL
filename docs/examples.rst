Examples
========

This page provides examples of using TwisteRL for reinforcement learning tasks.

Training Examples
-----------------

8-Puzzle Example
~~~~~~~~~~~~~~~~

The 8-puzzle is a classic sliding puzzle that consists of a 3x3 grid with numbered tiles and one empty space.

**Training:**

.. code-block:: bash

   python -m twisterl.train --config examples/ppo_puzzle8_v1.json

This example demonstrates:

- Fast training (under 1 minute on CPU)
- Discrete observation and action spaces
- PPO algorithm implementation

15-Puzzle Example
~~~~~~~~~~~~~~~~~

A more challenging 4x4 version of the sliding puzzle:

.. code-block:: bash

   python -m twisterl.train --config examples/ppo_puzzle15_v1.json

Inference Examples
------------------

Jupyter Notebooks
~~~~~~~~~~~~~~~~~

The ``examples/`` directory contains Jupyter notebooks for interactive exploration:

- **puzzle.ipynb**: Interactive example showing inference with trained puzzle models
- **hub_puzzle_model.ipynb**: Loading and using models from HuggingFace Hub

These notebooks demonstrate:

- Loading trained models
- Running inference
- Visualizing agent behavior

Creating Custom Environments
----------------------------

TwisteRL supports custom environments implemented in Rust. The ``examples/grid_world`` directory provides a complete working example.

**Steps to create a custom environment:**

1. **Create a new Rust crate:**

.. code-block:: bash

   cargo new --lib examples/my_env

2. **Add dependencies** in ``Cargo.toml``:

.. code-block:: toml

   [package]
   name = "my_env"
   version = "0.1.0"
   edition = "2021"

   [lib]
   name = "my_env"
   crate-type = ["cdylib"]

   [dependencies]
   pyo3 = { version = "0.20", features = ["extension-module"] }
   twisterl = { path = "path/to/twisterl/rust", features = ["python_bindings"] }

3. **Implement the environment** by implementing the ``twisterl::rl::env::Env`` trait.

4. **Expose it to Python** using ``PyBaseEnv``:

.. code-block:: rust

   use pyo3::prelude::*;
   use twisterl::python_interface::env::PyBaseEnv;

   #[pyclass(name = "MyEnv", extends = PyBaseEnv)]
   struct PyMyEnv;

   #[pymethods]
   impl PyMyEnv {
       #[new]
       fn new(...) -> (Self, PyBaseEnv) {
           let env = MyEnv::new(...);
           (PyMyEnv, PyBaseEnv { env: Box::new(env) })
       }
   }

5. **Build and install** the module:

.. code-block:: bash

   pip install .

6. **Use from Python** in a config file or directly.

See the `grid_world example <https://github.com/AI4quantum/twisteRL/tree/main/examples/grid_world>`_ for a complete implementation.

Python Environments
~~~~~~~~~~~~~~~~~~~

TwisteRL also supports Python environments through the ``PyEnv`` wrapper:

.. code-block:: json

   {
       "env_cls": "twisterl.envs.PyEnv",
       "env": {
           "pyenv_cls": "mymodule.MyPythonEnv"
       }
   }

Note that Python environments may be slower than native Rust environments.

Use Cases
---------

TwisteRL is particularly well-suited for:

**Quantum Circuit Transpilation**

Currently used in `Qiskit/qiskit-ibm-transpiler <https://github.com/Qiskit/qiskit-ibm-transpiler>`_ for AI-based circuit optimization including Clifford synthesis and routing.

**Puzzle-like Optimization Problems**

Problems with discrete state and action spaces where fast inference is important.

**Production-ready RL Inference**

Scenarios requiring high-performance inference with portable Rust-based models.
