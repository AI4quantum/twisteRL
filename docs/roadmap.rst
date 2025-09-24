Development Roadmap
===================

This page outlines the planned development roadmap for TwisteRL, from the current Proof of Concept to a full-featured production-ready framework.

Current State (v0.1.0 - PoC)
-----------------------------

**Status**: Proof of Concept ✅

**Key Features:**
- Hybrid Rust-Python implementation
- PPO and AlphaZero algorithms
- Discrete observation and action spaces  
- Native Rust environments and Python environment wrappers
- Basic training and inference capabilities

**Architecture:**
- Data collection and inference: Rust
- Training: Python (PyTorch)
- Environments: Rust + Python wrapper support

**Limitations:**
- Limited to discrete spaces and few available algorithms
- Minimal documentation and testing

Alpha Release (v0.2.0) - Q2 2025
---------------------------------

**Status**: In progress

**Upcoming Features (Alpha Version)**

- Full training in Rust
- Extended support for:
    - Continuous observation spaces
    - Continuous action spaces
    - Custom policy architectures
- Native WebAssembly environment support
- Streamlined policy+environment bundle export to WebAssembly
- Comprehensive Python interface
- Enhanced documentation and test coverage
- HuggingFace integration


Contributing to the Roadmap
---------------------------

**How to Influence Development:**

1. **GitHub Discussions**: Share your use cases and requirements
2. **Feature Requests**: Submit detailed feature requests with use cases
3. **Pull Requests**: Contribute implementations of roadmap features
4. **Community Feedback**: Participate in design discussions

**Priority Factors:**

- Community demand and use cases
- Technical feasibility and maintenance cost  
- Alignment with core mission (high-performance RL)
- Available contributor bandwidth

**Research Collaborations:**

We welcome collaborations with:

- Academic research groups
- Industry partners with interesting use cases
- Open source projects needing RL capabilities
- Hardware vendors (GPU, FPGA, specialized chips)

Stay Updated
------------

**Follow Development:**

- **GitHub**: Watch the repository for updates
- **Discussions**: Join GitHub Discussions for roadmap updates
- **Blog**: Development blog posts on major milestones
- **Twitter**: Follow @twisteRL for quick updates

**Release Notifications:**

- GitHub releases with detailed changelogs
- PyPI releases for easy installation
- Docker images for containerized deployment

This roadmap is a living document and will be updated based on community feedback and development progress. We're committed to transparent development and welcome your input on our direction! 🚀