Contributing to TwisteRL
=======================

We welcome contributions to TwisteRL! This guide will help you get started with contributing to the project.

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/AI4quantum/twisteRL.git
   cd twisteRL

3. Set up the development environment:

.. code-block:: bash

   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install in development mode
   pip install -e ".[dev]"

   # Install Rust toolchain (if not already installed)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env

4. Install pre-commit hooks:

.. code-block:: bash

   pre-commit install

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

The development environment includes additional tools:

- **Testing**: pytest, pytest-cov
- **Linting**: ruff, black, mypy
- **Documentation**: sphinx, sphinx-rtd-theme
- **Pre-commit**: pre-commit hooks for code quality

Code Style
----------

Python Code
~~~~~~~~~~~

We use several tools to maintain code quality:

- **ruff**: Fast Python linter and formatter
- **black**: Code formatter  
- **mypy**: Static type checker
- **isort**: Import sorting

Run code formatting:

.. code-block:: bash

   # Format Python code
   black src/ tests/
   ruff check src/ tests/ --fix
   isort src/ tests/

   # Type checking
   mypy src/

Rust Code
~~~~~~~~~

For Rust code, we follow standard Rust conventions:

.. code-block:: bash

   # Format Rust code
   cd rust/
   cargo fmt

   # Lint Rust code
   cargo clippy -- -D warnings

   # Run Rust tests
   cargo test

Testing
-------

Running Tests
~~~~~~~~~~~~~

We maintain comprehensive test suites for both Python and Rust components:

.. code-block:: bash

   # Run Python tests
   pytest tests/ -v --cov=src/

   # Run Rust tests
   cd rust/
   cargo test

   # Run integration tests
   pytest tests/integration/ -v

Writing Tests
~~~~~~~~~~~~~

**Python Tests:**

Place tests in the ``tests/`` directory with the ``test_`` prefix:

.. code-block:: python

   # tests/test_environments.py
   import pytest
   import twisterl

   def test_puzzle8_environment():
       env = twisterl.make_env("puzzle8_v1")
       obs = env.reset()
       
       assert obs.shape == (9,)
       assert env.action_space.n == 4
       
       action = env.action_space.sample()
       obs, reward, done, info = env.step(action)
       
       assert isinstance(reward, (int, float))
       assert isinstance(done, bool)

   @pytest.mark.parametrize("env_name", ["puzzle8_v1", "puzzle15_v1"])
   def test_environment_interface(env_name):
       env = twisterl.make_env(env_name)
       # Test common interface...

**Rust Tests:**

.. code-block:: rust

   // rust/src/envs/mod.rs
   #[cfg(test)]
   mod tests {
       use super::*;

       #[test]
       fn test_puzzle8_reset() {
           let mut env = Puzzle8::new();
           let obs = env.reset();
           assert_eq!(obs.len(), 9);
       }

       #[test] 
       fn test_puzzle8_step() {
           let mut env = Puzzle8::new();
           env.reset();
           let (obs, reward, done, _) = env.step(0);
           assert_eq!(obs.len(), 9);
       }
   }

Documentation
-------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd docs/
   make html

   # View documentation
   open _build/html/index.html  # macOS
   xdg-open _build/html/index.html  # Linux

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

- Use reStructuredText (.rst) format for documentation
- Include docstrings in all public functions and classes
- Add examples to complex functions
- Update the changelog for user-facing changes

**Python Docstring Style:**

.. code-block:: python

   def train_agent(config: dict) -> Agent:
       """Train a reinforcement learning agent.
       
       Args:
           config: Training configuration dictionary containing:
               - algorithm: Name of algorithm ("ppo", "alphazero")
               - environment: Name of environment
               - training: Training parameters
               
       Returns:
           Trained agent instance
           
       Example:
           >>> config = {"algorithm": "ppo", "environment": "puzzle8_v1"}
           >>> agent = train_agent(config)
           >>> action = agent.predict(observation)
       """

**Rust Documentation:**

.. code-block:: rust

   /// Train a PPO agent on the given environment.
   ///
   /// # Arguments
   /// 
   /// * `env` - The environment to train on
   /// * `config` - PPO configuration parameters
   ///
   /// # Returns
   ///
   /// A trained PPO agent
   ///
   /// # Example
   ///
   /// ```rust
   /// use twisterl::rl::PPO;
   /// let agent = PPO::train(&env, &config);
   /// ```
   pub fn train_ppo(env: &dyn Environment, config: &PPOConfig) -> PPO {
       // Implementation...
   }

Contribution Guidelines
-----------------------

Pull Request Process
~~~~~~~~~~~~~~~~~~~~

1. **Create a branch**: Use descriptive branch names

.. code-block:: bash

   git checkout -b feature/add-sac-algorithm
   git checkout -b fix/environment-reset-bug
   git checkout -b docs/improve-quickstart

2. **Make your changes**: Follow the code style guidelines

3. **Add tests**: Ensure your changes are tested

4. **Update documentation**: Add/update docs as needed

5. **Run the full test suite**:

.. code-block:: bash

   # Python tests and linting
   pytest tests/ -v
   ruff check src/ tests/
   black --check src/ tests/
   mypy src/

   # Rust tests and linting  
   cd rust/
   cargo test
   cargo clippy -- -D warnings
   cargo fmt --check

6. **Submit pull request**: Use the PR template

Pull Request Template
~~~~~~~~~~~~~~~~~~~~~

When submitting a PR, please include:

.. code-block:: text

   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature  
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   - [ ] Added tests for new functionality
   - [ ] All tests pass
   - [ ] Manual testing performed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Changelog updated (if applicable)

Issue Guidelines
~~~~~~~~~~~~~~~~

When reporting bugs or requesting features:

**Bug Reports:**
- Use the bug report template
- Include minimal reproduction case
- Specify environment details (OS, Python/Rust versions)
- Include error messages and stack traces

**Feature Requests:**  
- Clearly describe the motivation
- Provide example use cases
- Consider implementation complexity
- Discuss alternatives

Community Guidelines
--------------------

Please follow these guidelines:

- **Be respectful**: Treat all contributors with respect
- **Be constructive**: Provide helpful feedback
- **Be patient**: Reviews take time
- **Follow CoC**: Adhere to our Code of Conduct

**Code of Conduct highlights:**
- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community

Release Process
---------------

For maintainers, the release process is:

1. **Update version numbers** in ``pyproject.toml`` and ``Cargo.toml``
2. **Update changelog** with user-facing changes
3. **Create release PR** and get approval
4. **Tag release** and push to GitHub
5. **Publish to PyPI** (automated via GitHub Actions)
6. **Publish Rust crate** to crates.io


Thank you for contributing to TwisteRL! 🚀