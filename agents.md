# TwisteRL

A hybrid Rust/Python Reinforcement Learning framework optimized for speed.

## Project Overview

- **Rust (`rust/`)**: Core RL logic—environment stepping, trajectory collection, inference.
- **Python (`src/twisterl/`)**: Training loop, neural network optimization (PyTorch).

## Setup & Build Commands

```bash
# Install (editable mode, compiles Rust extension)
pip install -e .

# Rust check
cd rust && cargo check && cargo test

# Python tests
pytest tests/
```

## Code Style & Conventions

- **Rust**: Use `cargo fmt` and `cargo clippy` before committing.
- **Python**: PEP 8. Type hints encouraged.
- **Checkpoints**: Always use `safetensors` (never pickle `.pt`).
- **Symmetry**: Implement `TwistableEnv` for environments with symmetries.

## Testing Instructions

```bash
# Full Python test suite
pytest tests/

# Rust unit tests
cd rust && cargo test

# End-to-end training check
python -m twisterl.train --config examples/ppo_puzzle8_v1.json
```

## Directory Structure

```
twisteRL/
├── rust/              # Rust crate (core RL)
│   ├── src/           # Rust source
│   └── Cargo.toml     # Rust deps
├── src/twisterl/      # Python package
├── tests/             # Python tests
├── examples/          # Configs and notebooks
└── pyproject.toml     # Python build config
```

## Dev Environment Tips

- **Hybrid Build**: `pip install -e .` rebuilds the Rust extension.
- **Performance**: Prefer Rust for hot paths; Python for flexibility.
- **Traits**: `twisterl::rl::env::Env` is the core environment trait.
