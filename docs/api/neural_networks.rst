Neural Networks
===============

TwisteRL provides flexible neural network architectures optimized for reinforcement learning.

NN Module
---------

.. automodule:: twisterl.nn
   :members:
   :undoc-members:
   :show-inheritance:

Base Network Classes
--------------------

Network Base
~~~~~~~~~~~~

.. autoclass:: twisterl.nn.BaseNetwork
   :members:
   :undoc-members:
   :show-inheritance:

All neural networks in TwisteRL inherit from ``BaseNetwork`` and provide:

- Standardized initialization
- Device management (CPU/GPU)  
- Parameter counting and statistics
- Checkpointing support

Policy Networks
---------------

Actor Network
~~~~~~~~~~~~~

.. autoclass:: twisterl.nn.ActorNetwork
   :members:
   :undoc-members:
   :show-inheritance:

The actor network outputs action probabilities for discrete action spaces or action parameters for continuous spaces.

**Configuration:**

.. code-block:: python

   actor_config = {
       "hidden_layers": [256, 256],     # Hidden layer sizes
       "activation": "relu",            # Activation function
       "output_activation": "softmax",  # Output activation (discrete) 
       "dropout": 0.0,                  # Dropout probability
       "layer_norm": False,             # Use layer normalization
       "orthogonal_init": True          # Orthogonal weight initialization
   }

**Usage:**

.. code-block:: python

   import twisterl
   from twisterl.nn import ActorNetwork

   env = twisterl.make_env("puzzle8_v1")
   
   actor = ActorNetwork(
       input_dim=env.observation_space.shape[0],
       output_dim=env.action_space.n,
       hidden_layers=[128, 128],
       activation="relu"
   )

   # Forward pass
   import torch
   obs = torch.randn(32, env.observation_space.shape[0])  # Batch of observations
   action_probs = actor(obs)  # Shape: (32, action_space.n)

Critic Network
~~~~~~~~~~~~~~

.. autoclass:: twisterl.nn.CriticNetwork
   :members:
   :undoc-members:
   :show-inheritance:

The critic network estimates state values or action values (Q-functions).

**Configuration:**

.. code-block:: python

   critic_config = {
       "hidden_layers": [256, 256],
       "activation": "relu", 
       "output_activation": "linear",
       "dropout": 0.0,
       "layer_norm": False
   }

**Usage:**

.. code-block:: python

   from twisterl.nn import CriticNetwork

   critic = CriticNetwork(
       input_dim=env.observation_space.shape[0],
       output_dim=1,  # State value (V) or action value (Q)
       hidden_layers=[128, 128]
   )

   # Forward pass
   state_values = critic(obs)  # Shape: (32, 1)

Actor-Critic Networks
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: twisterl.nn.ActorCriticNetwork
   :members:
   :undoc-members:
   :show-inheritance:

Combined actor-critic architecture with shared feature layers:

.. code-block:: python

   from twisterl.nn import ActorCriticNetwork

   actor_critic = ActorCriticNetwork(
       input_dim=env.observation_space.shape[0],
       action_dim=env.action_space.n,
       shared_layers=[256, 256],      # Shared feature layers
       actor_layers=[128],            # Actor-specific layers  
       critic_layers=[128],           # Critic-specific layers
       activation="relu"
   )

   # Forward pass returns both policy and value
   obs = torch.randn(32, env.observation_space.shape[0])
   action_probs, state_values = actor_critic(obs)

AlphaZero Networks
------------------

Dual Head Network
~~~~~~~~~~~~~~~~~

.. autoclass:: twisterl.nn.AlphaZeroNetwork
   :members:
   :undoc-members:
   :show-inheritance:

AlphaZero uses a single network with two heads: policy and value.

.. code-block:: python

   from twisterl.nn import AlphaZeroNetwork

   alphazero_net = AlphaZeroNetwork(
       input_dim=env.observation_space.shape[0],
       action_dim=env.action_space.n,
       hidden_layers=[256, 256, 256],
       activation="relu",
       value_head_layers=[128],       # Value head architecture
       policy_head_layers=[128]       # Policy head architecture
   )

   # Forward pass
   obs = torch.randn(32, env.observation_space.shape[0])
   policy_logits, value = alphazero_net(obs)

Residual Networks
~~~~~~~~~~~~~~~~~

For complex environments, TwisteRL supports residual connections:

.. autoclass:: twisterl.nn.ResidualBlock
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from twisterl.nn import ResidualNetwork

   resnet = ResidualNetwork(
       input_dim=env.observation_space.shape[0],
       action_dim=env.action_space.n, 
       num_blocks=10,                 # Number of residual blocks
       block_size=256,                # Channels per block
       activation="relu"
   )

Convolutional Networks
----------------------

For image-based environments:

Conv2D Networks
~~~~~~~~~~~~~~~

.. autoclass:: twisterl.nn.ConvNetwork
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from twisterl.nn import ConvNetwork

   # For image observations (e.g., Atari)
   conv_net = ConvNetwork(
       input_shape=(4, 84, 84),       # (channels, height, width)
       action_dim=env.action_space.n,
       conv_layers=[
           {"filters": 32, "kernel_size": 8, "stride": 4},
           {"filters": 64, "kernel_size": 4, "stride": 2}, 
           {"filters": 64, "kernel_size": 3, "stride": 1}
       ],
       fc_layers=[512, 256]
   )

Nature DQN Architecture
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: twisterl.nn.NatureDQN
   :members:
   :undoc-members:
   :show-inheritance:

The standard CNN architecture from the DQN paper:

.. code-block:: python

   from twisterl.nn import NatureDQN

   nature_dqn = NatureDQN(
       input_shape=(4, 84, 84),
       action_dim=env.action_space.n
   )

Custom Architectures
--------------------

Building Custom Networks
~~~~~~~~~~~~~~~~~~~~~~~~~

You can easily create custom architectures by inheriting from ``BaseNetwork``:

.. code-block:: python

   import torch
   import torch.nn as nn
   from twisterl.nn import BaseNetwork

   class CustomNetwork(BaseNetwork):
       def __init__(self, input_dim, output_dim):
           super().__init__()
           
           self.feature_extractor = nn.Sequential(
               nn.Linear(input_dim, 256),
               nn.ReLU(),
               nn.Linear(256, 256), 
               nn.ReLU()
           )
           
           self.policy_head = nn.Linear(256, output_dim)
           self.value_head = nn.Linear(256, 1)
           
           # Initialize weights
           self.apply(self._init_weights)
       
       def forward(self, x):
           features = self.feature_extractor(x)
           policy = torch.softmax(self.policy_head(features), dim=-1)
           value = self.value_head(features)
           return policy, value
       
       def _init_weights(self, module):
           if isinstance(module, nn.Linear):
               torch.nn.init.orthogonal_(module.weight)
               module.bias.data.zero_()

Attention Networks
~~~~~~~~~~~~~~~~~~

For environments with variable-length or structured inputs:

.. autoclass:: twisterl.nn.AttentionNetwork
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from twisterl.nn import AttentionNetwork

   attention_net = AttentionNetwork(
       input_dim=256,
       num_heads=8,                   # Multi-head attention
       hidden_dim=512,                # Feed-forward hidden dimension  
       num_layers=6,                  # Number of attention layers
       dropout=0.1
   )

Network Utilities
-----------------

Initialization
~~~~~~~~~~~~~~

.. autofunction:: twisterl.nn.init_weights
.. autofunction:: twisterl.nn.orthogonal_init
.. autofunction:: twisterl.nn.xavier_init

.. code-block:: python

   from twisterl.nn import init_weights, orthogonal_init

   # Initialize network weights
   network = ActorNetwork(input_dim=100, output_dim=10)
   network.apply(orthogonal_init)

   # Or use the utility function
   init_weights(network, gain=1.0)

Activation Functions
~~~~~~~~~~~~~~~~~~~

.. autofunction:: twisterl.nn.get_activation
.. autofunction:: twisterl.nn.Swish
.. autofunction:: twisterl.nn.Mish

.. code-block:: python

   from twisterl.nn import get_activation

   # Get activation function by name
   activation = get_activation("relu")      # Returns nn.ReLU()
   activation = get_activation("swish")     # Returns Swish()
   activation = get_activation("mish")      # Returns Mish()

Layer Utilities
~~~~~~~~~~~~~~~

.. autoclass:: twisterl.nn.MLP
   :members:
   :show-inheritance:

.. autoclass:: twisterl.nn.LayerNorm
   :members:
   :show-inheritance:

.. code-block:: python

   from twisterl.nn import MLP

   # Quick MLP creation
   mlp = MLP(
       input_dim=100,
       output_dim=10, 
       hidden_layers=[256, 256],
       activation="relu",
       dropout=0.1,
       layer_norm=True
   )

Performance Optimizations
-------------------------

Model Compilation
~~~~~~~~~~~~~~~~~

For PyTorch 2.0+ users, TwisteRL supports model compilation:

.. code-block:: python

   import torch
   from twisterl.nn import ActorNetwork

   # Create network
   actor = ActorNetwork(input_dim=100, output_dim=10)

   # Compile for faster inference (PyTorch 2.0+)
   if hasattr(torch, 'compile'):
       actor = torch.compile(actor)

Mixed Precision Training
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torch.cuda.amp import autocast, GradScaler

   # Enable mixed precision
   scaler = GradScaler()
   
   # In training loop
   with autocast():
       policy, value = actor_critic(obs)
       loss = compute_loss(policy, value, targets)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()

Device Management
~~~~~~~~~~~~~~~~~

.. autofunction:: twisterl.nn.get_device
.. autofunction:: twisterl.nn.move_to_device

.. code-block:: python

   from twisterl.nn import get_device, move_to_device

   # Automatic device selection
   device = get_device()  # Returns "cuda" if available, else "cpu"

   # Move tensors/models to device
   model = move_to_device(model, device)
   obs = move_to_device(obs, device)

Model Statistics
----------------

.. autofunction:: twisterl.nn.count_parameters
.. autofunction:: twisterl.nn.model_summary

.. code-block:: python

   from twisterl.nn import count_parameters, model_summary

   # Count parameters
   total_params = count_parameters(network)
   print(f"Total parameters: {total_params:,}")

   # Detailed model summary  
   model_summary(network, input_shape=(100,))

This will output something like:

.. code-block:: text

   ================================================================
   Layer (type:depth-idx)                   Param #
   ================================================================
   ActorNetwork                             --
   ├─Sequential: 1-1                        --
   │    └─Linear: 2-1                       25,856
   │    └─ReLU: 2-2                         --
   │    └─Linear: 2-3                       65,792  
   │    └─ReLU: 2-4                         --
   ├─Linear: 1-2                            2,570
   ================================================================
   Total params: 94,218
   Trainable params: 94,218
   Non-trainable params: 0