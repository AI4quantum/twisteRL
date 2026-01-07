<p align="center">
  <img src="./assets/twisterl-logo.png" width="200" alt="TwisteRL"/>
</p>

# TwisteRL

A minimalistic, high-performance Reinforcement Learning framework implemented in **Rust** and **Mojo**.

The current version is a *Proof of Concept*, stay tuned for future releases!

## Implementations

| Language | Location | Status |
|----------|----------|--------|
| Rust | `rust/` | Core implementation with Python bindings |
| Mojo | `mojo/` | Optimized port with competitive performance |

## Install

### Python (Rust backend)
```shell
pip install .
```

### Mojo
```shell
cd mojo
# Requires Mojo 25.5+ from https://docs.modular.com/mojo/manual/get-started
```

## Use

### Training (Python/Rust)
```shell
python -m twisterl.train --config examples/ppo_puzzle8_v1.json
```

### Training (Mojo)
```shell
cd mojo
mojo run train_puzzle.mojo
```
This example trains a model to play the popular "8 puzzle":

```
|8|7|5|
|3|2| |
|4|6|1|
```

where numbers have to be shifted around through the empty slot until they are in order.

This model can be trained on a single CPU in under 1 minute (no GPU required!). 
A larger version (4x4) is available: `examples/ppo_puzzle15_v1.json`.


### Inference
Check the notebook example [here](examples/puzzle.ipynb)!


## 🚀 Key Features
- **High-Performance Core**: RL episode loop implemented in Rust and Mojo for faster training and inference
- **Inference-Ready**: Easy compilation and bundling of models with environments into portable binaries for inference
- **Modular Design**: Support for multiple algorithms (PPO, AlphaZero, Evolution Strategies) with interchangeable training and inference
- **Language Interoperability**: Core in Rust/Mojo with Python interface

## ⚡ Performance (Mojo vs Rust)

Benchmark results on 8-puzzle environment (100K iterations):

| Operation | Mojo | Rust | Winner |
|-----------|------|------|--------|
| Reset | 0.06 μs | 0.13 μs | Mojo 2.2x faster |
| Combined RL step | 0.07 μs | 0.28 μs | Mojo 4x faster |
| Episode rollout | 0.09 ms/10K | 0.53 ms/1K | Mojo 5.8x faster |

Run benchmarks:
```shell
# Mojo
cd mojo && mojo run benchmark.mojo

# Rust
cd rust && cargo run --release --bin benchmark
```


## 🏗️ Current State (PoC)

### Rust Implementation
- Hybrid rust-python implementation:
    - Data collection and inference in Rust
    - Training in Python (PyTorch)
- Supported algorithms: PPO, AlphaZero
- Support for native Rust environments and Python environments through a wrapper

### Mojo Implementation
- Pure Mojo implementation with no external dependencies
- Training uses Evolution Strategies (derivative-free, no backprop needed)
- Optimized with `InlineArray` for stack allocation and `@always_inline`
- Model save/load support

### Common
- Focus on discrete observation and action spaces


## 🚧 Roadmap
Upcoming Features (Alpha Version)

- Full training in Rust (without PyTorch dependency)
- Mojo GPU acceleration with MAX
- Extended support for:
    - Continuous observation spaces
    - Continuous action spaces
    - Custom policy architectures
- Native WebAssembly environment support
- Streamlined policy+environment bundle export to WebAssembly
- Comprehensive Python interface
- Enhanced documentation and test coverage

## 💎 Future Possibilities

- WebAssembly environment repository
- Browser-based environment and agent visualization
- Interactive web demonstrations
- Serverless distributed training

## 🎮 Use Cases

Currently used in:

- Qiskit Quantum circuit transpiling AI models (Clifford synthesis, routing) [Qiskit/qiskit-ibm-transpiler ](https://github.com/Qiskit/qiskit-ibm-transpiler)

Perfect for:
- Puzzle-like optimization problems
- Any scenario requiring fast, production performance RL inference

## 🔧 Current Limitations

- Limited to discrete observation and action spaces
- Python environments may create performance bottlenecks (Rust)
- Mojo version currently supports Evolution Strategies only (no PPO/AlphaZero yet)
- Documentation and testing coverage is currently minimal
- WebAssembly support is in development

## 🤝 Contributing

We're in early development stages and welcome contributions! Stay tuned for more detailed contribution guidelines.

##  📄 Note

This project is currently in PoC stage. While functional, it's under active development and the API may change significantly.

## 📜 License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0