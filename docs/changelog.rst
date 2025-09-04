Changelog
=========

All notable changes to TwisteRL will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
~~~~~
- Comprehensive Sphinx documentation with API reference
- GitHub Pages documentation deployment
- Enhanced project structure with proper Python packaging

Changed
~~~~~~~
- Improved README with clearer installation and usage instructions

[0.1.0] - 2024-12-XX
---------------------

Added
~~~~~

**Core Features**
- Initial Proof of Concept release
- Hybrid Rust-Python implementation for high-performance RL
- PPO (Proximal Policy Optimization) algorithm implementation
- AlphaZero algorithm implementation with MCTS
- Support for discrete observation and action spaces
- Native Rust environments with Python wrapper support

**Environments**
- Puzzle8 environment (3x3 sliding puzzle)
- Puzzle15 environment (4x4 sliding puzzle) 
- Base environment interface for custom environments
- Python environment wrapper system

**Training System**
- Command-line training interface (``python -m twisterl.train``)
- JSON-based configuration system
- TensorBoard integration for training monitoring
- Model checkpointing and saving/loading

**Neural Networks**
- PyTorch-based neural network implementations
- Actor-critic architecture for PPO
- Dual-head network for AlphaZero (policy + value)
- Configurable network architectures

**Python Package**
- Proper Python packaging with ``pyproject.toml``
- Maturin-based Rust-Python bindings
- Installation via pip

**Examples and Documentation**
- Training configuration examples
- Jupyter notebook demonstration (``examples/puzzle.ipynb``)
- Basic README with quickstart guide

**Development Infrastructure**
- GitHub Actions CI/CD pipeline
- Python and Rust unit tests
- Code linting and formatting
- Pre-commit hooks

Technical Details
~~~~~~~~~~~~~~~~~

**Performance Characteristics**
- Training: ~10,000 timesteps/second (single CPU core)
- Inference: ~50,000 predictions/second
- Memory efficient episode collection in Rust
- Zero-copy data transfer between Rust and Python where possible

**Supported Platforms**
- Linux x86_64
- macOS (Intel and Apple Silicon)
- Windows x86_64

**Dependencies**
- Python 3.9+
- PyTorch 2.2+
- NumPy 2.0+
- Rust 1.70+ (for building from source)

Known Issues
~~~~~~~~~~~~
- Limited to discrete observation and action spaces
- Python training creates performance bottlenecks
- Documentation and test coverage is minimal
- No WebAssembly support yet
- No continuous action space support

Use Cases
~~~~~~~~~

Currently being used in:
- Qiskit quantum circuit transpilation (IBM Quantum)
- Puzzle-solving and combinatorial optimization research
- RL algorithm prototyping and benchmarking

[0.0.1] - 2024-06-XX (Internal)
--------------------------------

Added
~~~~~
- Initial project structure
- Basic Rust environment framework
- Minimal PPO implementation
- Puzzle8 environment prototype

---

**Legend:**
- ``Added`` for new features
- ``Changed`` for changes in existing functionality  
- ``Deprecated`` for soon-to-be removed features
- ``Removed`` for now removed features
- ``Fixed`` for any bug fixes
- ``Security`` for vulnerability fixes