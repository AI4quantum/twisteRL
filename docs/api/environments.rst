Environments
============

TwisteRL provides a flexible environment system supporting both native Rust environments and Python environment wrappers.

Base Environment Interface
---------------------------

.. automodule:: twisterl.envs
   :members:
   :undoc-members:
   :show-inheritance:

Built-in Environments
----------------------

Puzzle Environments
~~~~~~~~~~~~~~~~~~~~

TwisteRL includes several built-in puzzle environments perfect for testing and development:

**Puzzle8 (puzzle8_v1)**

The classic 8-puzzle sliding tile game:

- **Observation Space**: Flattened 3x3 grid (9 integers)
- **Action Space**: Discrete(4) - Up, Down, Left, Right  
- **Reward**: -1 per step, +100 for solving
- **Episode Length**: Maximum 200 steps

**Puzzle15 (puzzle15_v1)**

The larger 15-puzzle variant:

- **Observation Space**: Flattened 4x4 grid (16 integers)
- **Action Space**: Discrete(4) - Up, Down, Left, Right
- **Reward**: -1 per step, +100 for solving  
- **Episode Length**: Maximum 500 steps

Environment Factory
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import twisterl

   # Create built-in environment
   env = twisterl.make_env("puzzle8_v1")

   # Environment info
   print(f"Observation space: {env.observation_space}")
   print(f"Action space: {env.action_space}")

Custom Environments
-------------------

Creating Rust Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For maximum performance, implement environments directly in Rust:

.. code-block:: rust

   use twisterl::envs::{Environment, ObservationSpace, ActionSpace};

   pub struct CustomEnv {
       state: Vec<i32>,
   }

   impl Environment for CustomEnv {
       type Observation = Vec<i32>;
       type Action = i32;

       fn reset(&mut self) -> Self::Observation {
           self.state = vec![0; 10];
           self.state.clone()
       }

       fn step(&mut self, action: Self::Action) -> (Self::Observation, f64, bool, serde_json::Value) {
           // Environment logic here
           let reward = 0.0;
           let done = false;
           let info = serde_json::json!({});
           (self.state.clone(), reward, done, info)
       }

       fn observation_space(&self) -> ObservationSpace {
           ObservationSpace::Box { low: vec![0], high: vec![100], shape: vec![10] }
       }

       fn action_space(&self) -> ActionSpace {
           ActionSpace::Discrete { n: 4 }
       }
   }

Creating Python Environments  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For rapid prototyping, create Python environments:

.. code-block:: python

   import twisterl
   import numpy as np

   class CustomPythonEnv:
       def __init__(self):
           self.state = None
           
       def reset(self):
           self.state = np.zeros(10)
           return self.state.copy()
           
       def step(self, action):
           # Environment logic
           reward = 0.0
           done = False
           info = {}
           return self.state.copy(), reward, done, info
           
       @property 
       def observation_space(self):
           return twisterl.spaces.Box(low=0, high=100, shape=(10,))
           
       @property
       def action_space(self):
           return twisterl.spaces.Discrete(4)

   # Register custom environment
   twisterl.register_env("custom_env_v1", CustomPythonEnv)

   # Use it
   env = twisterl.make_env("custom_env_v1")

Environment Wrappers
--------------------

TwisteRL provides several wrappers to modify environment behavior:

Observation Wrappers
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from twisterl.envs.wrappers import ObservationWrapper

   class NormalizeObservation(ObservationWrapper):
       def observation(self, obs):
           return obs / 255.0  # Normalize to [0,1]

   env = twisterl.make_env("puzzle8_v1")
   env = NormalizeObservation(env)

Reward Wrappers
~~~~~~~~~~~~~~~

.. code-block:: python

   from twisterl.envs.wrappers import RewardWrapper

   class ScaleReward(RewardWrapper):
       def __init__(self, env, scale=0.01):
           super().__init__(env)
           self.scale = scale
           
       def reward(self, reward):
           return reward * self.scale

   env = ScaleReward(env, scale=0.01)

Action Wrappers
~~~~~~~~~~~~~~~

.. code-block:: python

   from twisterl.envs.wrappers import ActionWrapper

   class DiscreteToBox(ActionWrapper):
       def action(self, action):
           # Convert discrete to continuous
           return float(action) / (self.action_space.n - 1)

Environment Spaces
------------------

TwisteRL supports standard Gym-style spaces:

Discrete Spaces
~~~~~~~~~~~~~~~

.. code-block:: python

   from twisterl.spaces import Discrete

   # 4 possible actions: 0, 1, 2, 3
   action_space = Discrete(4)

Box Spaces  
~~~~~~~~~~

.. code-block:: python

   from twisterl.spaces import Box
   import numpy as np

   # Continuous space [0, 1]^10
   obs_space = Box(low=0.0, high=1.0, shape=(10,))

   # With different bounds per dimension
   obs_space = Box(
       low=np.array([0, -1, 0]), 
       high=np.array([1, 1, 10])
   )

Multi-Discrete Spaces
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from twisterl.spaces import MultiDiscrete

   # Two discrete choices: first has 3 options, second has 5
   action_space = MultiDiscrete([3, 5])

Performance Considerations
--------------------------

**Rust vs Python Environments:**

- **Rust environments**: 10-100x faster, recommended for production
- **Python environments**: Easier to prototype, good for development

**Environment Design Tips:**

1. **Minimize observation copying**: Return views when possible
2. **Batch operations**: Vectorize computations in Python
3. **Avoid dynamic allocation**: Pre-allocate arrays in tight loops
4. **Profile bottlenecks**: Use profiling tools to identify slow parts

**Benchmarking:**

.. code-block:: python

   import time
   import twisterl

   def benchmark_env(env_name, num_steps=10000):
       env = twisterl.make_env(env_name)
       
       start = time.time()
       obs = env.reset()
       
       for _ in range(num_steps):
           action = env.action_space.sample()
           obs, reward, done, info = env.step(action)
           if done:
               obs = env.reset()
       
       end = time.time()
       print(f"{env_name}: {num_steps/(end-start):.0f} steps/second")

   benchmark_env("puzzle8_v1")  # Rust environment
   benchmark_env("custom_python_env")  # Python environment