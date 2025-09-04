Examples
========

This page provides comprehensive examples of using TwisteRL for various reinforcement learning tasks.

Training Examples
-----------------

8-Puzzle Example
~~~~~~~~~~~~~~~~

The 8-puzzle is a classic sliding puzzle that consists of a 3x3 grid with numbered tiles and one empty space.

**Configuration (examples/ppo_puzzle8_v1.json):**

.. code-block:: bash

   python -m twisterl.train --config examples/ppo_puzzle8_v1.json

This example demonstrates:

- Fast training (under 1 minute on CPU)
- Discrete observation and action spaces
- PPO algorithm implementation

15-Puzzle Example
~~~~~~~~~~~~~~~~~

A more challenging 4x4 version of the sliding puzzle:

.. code-block:: bash

   python -m twisterl.train --config examples/ppo_puzzle15_v1.json

Inference Examples
------------------

Jupyter Notebook
~~~~~~~~~~~~~~~~

The `examples/puzzle.ipynb` notebook provides an interactive example showing:

- Loading trained models
- Running inference
- Visualizing agent behavior
- Performance analysis

Python Script Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import twisterl
   import numpy as np

   def run_episode(agent, env, render=False):
       """Run a single episode and return the total reward."""
       obs = env.reset()
       total_reward = 0
       steps = 0
       
       while True:
           if render:
               env.render()
           
           # Get action from agent
           action = agent.predict(obs)
           
           # Take step in environment
           obs, reward, done, info = env.step(action)
           total_reward += reward
           steps += 1
           
           if done:
               break
       
       return total_reward, steps

   # Load trained agent
   agent = twisterl.load_agent("models/ppo_puzzle8.pt")
   
   # Create environment
   env = twisterl.make_env("puzzle8_v1")
   
   # Run multiple episodes
   rewards = []
   for episode in range(100):
       reward, steps = run_episode(agent, env)
       rewards.append(reward)
       print(f"Episode {episode + 1}: Reward = {reward}, Steps = {steps}")
   
   print(f"Average reward: {np.mean(rewards):.2f}")

Custom Environment Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import twisterl
   from twisterl.envs import BaseEnvironment

   class CustomGridWorld(BaseEnvironment):
       """A simple grid world environment."""
       
       def __init__(self, size=5):
           self.size = size
           self.agent_pos = [0, 0]
           self.goal_pos = [size-1, size-1]
           
       def reset(self):
           self.agent_pos = [0, 0]
           return self._get_observation()
       
       def step(self, action):
           # 0: up, 1: down, 2: left, 3: right
           moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
           dx, dy = moves[action]
           
           new_x = max(0, min(self.size-1, self.agent_pos[0] + dx))
           new_y = max(0, min(self.size-1, self.agent_pos[1] + dy))
           self.agent_pos = [new_x, new_y]
           
           reward = 1.0 if self.agent_pos == self.goal_pos else -0.1
           done = self.agent_pos == self.goal_pos
           
           return self._get_observation(), reward, done, {}
       
       def _get_observation(self):
           return self.agent_pos + self.goal_pos

   # Use custom environment
   env = CustomGridWorld(size=10)
   # Train agent with this environment...

Performance Benchmarking
-------------------------

.. code-block:: python

   import time
   import twisterl

   def benchmark_training():
       """Benchmark training performance."""
       start_time = time.time()
       
       # Configure training
       config = {
           "algorithm": "ppo",
           "environment": "puzzle8_v1",
           "training": {
               "total_timesteps": 50000,
               "learning_rate": 0.0003
           }
       }
       
       # Run training
       agent = twisterl.train(config)
       
       end_time = time.time()
       training_time = end_time - start_time
       
       print(f"Training completed in {training_time:.2f} seconds")
       print(f"Timesteps per second: {50000/training_time:.0f}")
       
       return agent

   def benchmark_inference(agent, num_episodes=1000):
       """Benchmark inference performance."""
       env = twisterl.make_env("puzzle8_v1")
       
       start_time = time.time()
       total_steps = 0
       
       for _ in range(num_episodes):
           obs = env.reset()
           done = False
           while not done:
               action = agent.predict(obs)
               obs, _, done, _ = env.step(action)
               total_steps += 1
       
       end_time = time.time()
       inference_time = end_time - start_time
       
       print(f"Inference: {total_steps/inference_time:.0f} steps/second")

Use Cases
---------

TwisteRL is particularly well-suited for:

**Training new models for AI-based transpilation**: Clifford synthesis, routing using https://github.com/AI4quantum/qiskit-gym

**Puzzle-like Optimization Problems**

**Production-ready RL Inference**
