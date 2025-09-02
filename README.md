<p align="center">
  <img src="./assets/twisterl-logo.png" width="200" alt="TwisteRL"/>
</p>

# TwisteRL

A minimalistic, high-performance Reinforcement Learning framework implemented in Rust.

The current version is a *Proof of Concept*, stay tuned for future releases!

## Install

```shell
pip install twisterl
```

## Use

### Training
```shell
python -m twisterl.train --config examples/ppo_puzzle8_v1.json
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
- **High-Performance Core**: RL episode loop implemented in Rust for faster training and inference
- **Inference-Ready**: Easy compilation and bundling of models with environments into portable binaries for inference
- **Modular Design**: Support for multiple algorithms (PPO, AlphaZero) with interchangeable training and inference
- **Language Interoperability**: Core in Rust with Python interface


## 🏗️ Current State (PoC)
- Hybrid rust-python implementation:
    - Data collection and inference in Rust
    - Training in Python (PyTorch)
- Supported algorithms:
    - PPO (Proximal Policy Optimization)
    - AlphaZero
- Focus on discrete observation and action spaces
- Support for native Rust environments and for Python environments through a wrapper


## 🚧 Roadmap
Upcoming Features (Alpha Version)

- Full training in Rust
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
- Python environments may create performance bottlenecks
- Documentation and testing coverage is currently minimal
- WebAssembly support is in development

## 🤝 Contributing

We're in early development stages and welcome contributions! Stay tuned for more detailed contribution guidelines.

##  📄 Note

This project is currently in PoC stage. While functional, it's under active development and the API may change significantly.

## 📜 License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0