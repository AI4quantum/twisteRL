Neural Networks
===============

TwisteRL provides neural network architectures for reinforcement learning policies.

NN Module
---------

.. automodule:: twisterl.nn
   :members:
   :undoc-members:
   :show-inheritance:

Policy Networks
---------------

BasicPolicy
~~~~~~~~~~~

.. autoclass:: twisterl.nn.BasicPolicy
   :members:
   :undoc-members:
   :show-inheritance:

The ``BasicPolicy`` is the main policy network used for both PPO and AlphaZero. It has an actor-critic architecture with shared embedding layers.

**Configuration:**

.. code-block:: json

   {
       "policy_cls": "twisterl.nn.BasicPolicy",
       "policy": {
           "embedding_size": 512,
           "common_layers": [256],
           "policy_layers": [],
           "value_layers": []
       }
   }

**Parameters:**

- **embedding_size**: Size of the embedding layer
- **common_layers**: Hidden layer sizes for shared network
- **policy_layers**: Additional layers for policy head (after common layers)
- **value_layers**: Additional layers for value head (after common layers)

**Architecture:**

1. Embedding layer: ``obs_size -> embedding_size`` (Linear + ReLU)
2. Common layers: Shared MLP
3. Policy head: Outputs action logits
4. Value head: Outputs state value

**Usage:**

.. code-block:: python

   from twisterl.nn import BasicPolicy

   policy = BasicPolicy(
       obs_shape=[9],           # 3x3 puzzle = 9 observations
       num_actions=4,           # 4 possible moves
       embedding_size=512,
       common_layers=(256,),
       policy_layers=(),
       value_layers=(),
       obs_perms=(),            # Observation permutations (twists)
       act_perms=()             # Action permutations (twists)
   )

   # Forward pass (returns logits, not probabilities)
   import torch
   obs = torch.randn(32, 9)
   logits, values = policy(obs)

   # Predict with numpy input (returns action probabilities and value)
   action_probs, value = policy.predict(obs_numpy)

Conv1dPolicy
~~~~~~~~~~~~

.. autoclass:: twisterl.nn.Conv1dPolicy
   :members:
   :undoc-members:
   :show-inheritance:

A variant of BasicPolicy that uses 1D convolutions for the embedding layer. Useful for environments with structured 2D observations.

**Parameters:**

- **conv_dim**: Which dimension to convolve over (0 or 1)

Permutation Support (Twists)
----------------------------

Both policy classes support permutation symmetries ("twists") for symmetry-aware training:

.. code-block:: python

   # Get twists from environment
   obs_perms, act_perms = env.twists()

   # Create policy with permutation support
   policy = BasicPolicy(
       obs_shape=env.obs_shape(),
       num_actions=env.num_actions(),
       obs_perms=obs_perms,
       act_perms=act_perms,
       ...
   )

When permutations are provided, the policy can:

- Apply random permutations during training for data augmentation
- Handle permutation indices passed during forward pass

Rust Conversion
---------------

Policies can be converted to Rust for fast inference:

.. code-block:: python

   # Convert PyTorch policy to Rust
   rust_policy = policy.to_rust()

This is used internally during training for fast data collection.

Network Utilities
-----------------

.. automodule:: twisterl.nn.utils
   :members:
   :undoc-members:
   :show-inheritance:

Key utility functions:

- ``make_sequential(in_size, layer_sizes, final_relu=True)``: Create a sequential MLP
- ``sequential_to_rust(module)``: Convert PyTorch Sequential to Rust
- ``embeddingbag_to_rust(module, shape, dim)``: Convert embedding layer to Rust

Device Management
-----------------

Policies automatically handle device placement:

.. code-block:: python

   # Move to GPU if available
   policy = policy.to("cuda")
   policy.device = "cuda"

   # Or use config-based device selection
   # (handled automatically by Algorithm class)
