Contributing to TwisteRL
=======================

We welcome contributions to TwisteRL! This guide will help you get started.

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/twisteRL.git
   cd twisteRL

3. Set up the development environment:

.. code-block:: bash

   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install Rust toolchain (if not already installed)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env

   # Install the package (builds Rust extension)
   pip install -e .

Code Style
----------

Python Code
~~~~~~~~~~~

Follow standard Python conventions. Format code before committing.

Rust Code
~~~~~~~~~

For Rust code, follow standard Rust conventions:

.. code-block:: bash

   # Format Rust code
   cargo fmt

   # Lint Rust code
   cargo clippy -- -D warnings

   # Run Rust tests
   cargo test

Testing
-------

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run Python tests
   pytest tests/ -v

   # Run Rust tests
   cargo test

Documentation
-------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd docs/
   pip install -r requirements.txt
   make html

   # View documentation
   open _build/html/index.html  # macOS
   xdg-open _build/html/index.html  # Linux

Contribution Guidelines
-----------------------

Pull Request Process
~~~~~~~~~~~~~~~~~~~~

1. **Create a branch**: Use descriptive branch names

.. code-block:: bash

   git checkout -b feature/my-feature
   git checkout -b fix/bug-description

2. **Make your changes**: Follow the code style guidelines

3. **Add tests**: Ensure your changes are tested

4. **Update documentation**: Add/update docs as needed

5. **Submit pull request**

Issue Guidelines
~~~~~~~~~~~~~~~~

**Bug Reports:**

- Include minimal reproduction case
- Specify environment details (OS, Python version)
- Include error messages and stack traces

**Feature Requests:**

- Clearly describe the motivation
- Provide example use cases

Community Guidelines
--------------------

- Be respectful and constructive
- Follow the Code of Conduct
- Be patient with reviews

Thank you for contributing to TwisteRL!
