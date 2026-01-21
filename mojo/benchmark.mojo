# -*- coding: utf-8 -*-
# (C) Copyright 2025 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0.

"""
Benchmark for TwisterL Mojo implementation.
Compares performance with the Rust implementation.

Optimized version using InlineArray for stack-allocated data.
"""

from time import perf_counter_ns
from collections import List

from twisterl.envs.puzzle import PuzzleEnv, NUM_ACTIONS, PUZZLE_SIZE


alias NUM_ITERATIONS = 100000  # Increased for more accurate timing
alias NUM_EPISODES = 10000


fn benchmark_env_operations():
    print("\n--- Environment Operations Benchmark (Optimized) ---")

    # Benchmark: Create environment
    var start = perf_counter_ns()
    for _ in range(NUM_ITERATIONS):
        var env = PuzzleEnv(3, 3, 5, 2, 20)
        _ = env  # Prevent optimization
    var elapsed = perf_counter_ns() - start
    var elapsed_us = Float64(elapsed) / 1000.0
    print("Create env (", NUM_ITERATIONS, " iterations):", elapsed_us / 1000.0, "ms (", elapsed_us / NUM_ITERATIONS, "us/iter)")

    # Benchmark: Reset
    var env = PuzzleEnv(3, 3, 5, 2, 20)
    start = perf_counter_ns()
    for _ in range(NUM_ITERATIONS):
        env.reset()
    elapsed = perf_counter_ns() - start
    elapsed_us = Float64(elapsed) / 1000.0
    print("Reset (", NUM_ITERATIONS, " iterations):", elapsed_us / 1000.0, "ms (", elapsed_us / NUM_ITERATIONS, "us/iter)")

    # Benchmark: Step
    env = PuzzleEnv(3, 3, 5, 2, 20)
    env.reset()
    start = perf_counter_ns()
    for i in range(NUM_ITERATIONS):
        env.step(i % 4)
    elapsed = perf_counter_ns() - start
    elapsed_us = Float64(elapsed) / 1000.0
    print("Step (", NUM_ITERATIONS, " iterations):", elapsed_us / 1000.0, "ms (", elapsed_us / NUM_ITERATIONS, "us/iter)")

    # Benchmark: Observe (optimized InlineArray version)
    env = PuzzleEnv(3, 3, 5, 2, 20)
    start = perf_counter_ns()
    for _ in range(NUM_ITERATIONS):
        var obs = env.observe()
        _ = obs
    elapsed = perf_counter_ns() - start
    elapsed_us = Float64(elapsed) / 1000.0
    print("Observe (", NUM_ITERATIONS, " iterations):", elapsed_us / 1000.0, "ms (", elapsed_us / NUM_ITERATIONS, "us/iter)")

    # Benchmark: Masks (optimized InlineArray version)
    env = PuzzleEnv(3, 3, 5, 2, 20)
    start = perf_counter_ns()
    for _ in range(NUM_ITERATIONS):
        var masks = env.masks()
        _ = masks
    elapsed = perf_counter_ns() - start
    elapsed_us = Float64(elapsed) / 1000.0
    print("Masks (", NUM_ITERATIONS, " iterations):", elapsed_us / 1000.0, "ms (", elapsed_us / NUM_ITERATIONS, "us/iter)")

    # Benchmark: Clone (optimized - direct InlineArray copy)
    env = PuzzleEnv(3, 3, 5, 2, 20)
    start = perf_counter_ns()
    for _ in range(NUM_ITERATIONS):
        var cloned = env.clone()
        _ = cloned
    elapsed = perf_counter_ns() - start
    elapsed_us = Float64(elapsed) / 1000.0
    print("Clone (", NUM_ITERATIONS, " iterations):", elapsed_us / 1000.0, "ms (", elapsed_us / NUM_ITERATIONS, "us/iter)")

    # Benchmark: is_final
    env = PuzzleEnv(3, 3, 5, 2, 20)
    start = perf_counter_ns()
    for _ in range(NUM_ITERATIONS):
        var is_final = env.is_final()
        _ = is_final
    elapsed = perf_counter_ns() - start
    elapsed_us = Float64(elapsed) / 1000.0
    print("is_final (", NUM_ITERATIONS, " iterations):", elapsed_us / 1000.0, "ms (", elapsed_us / NUM_ITERATIONS, "us/iter)")

    # Benchmark: reward
    env = PuzzleEnv(3, 3, 5, 2, 20)
    start = perf_counter_ns()
    for _ in range(NUM_ITERATIONS):
        var reward = env.reward()
        _ = reward
    elapsed = perf_counter_ns() - start
    elapsed_us = Float64(elapsed) / 1000.0
    print("Reward (", NUM_ITERATIONS, " iterations):", elapsed_us / 1000.0, "ms (", elapsed_us / NUM_ITERATIONS, "us/iter)")


fn benchmark_episode_rollout():
    print("\n--- Episode Rollout Benchmark (Optimized) ---")

    var start = perf_counter_ns()
    for _ in range(NUM_EPISODES):
        var env = PuzzleEnv(3, 3, 5, 2, 20)
        env.reset()

        var step_count = 0
        while not env.is_final() and step_count < 100:
            var masks = env.masks()
            # Simple policy: pick first valid action
            var action = 0
            for i in range(NUM_ACTIONS):
                if masks[i]:
                    action = i
                    break
            env.step(action)
            var obs = env.observe()
            var reward = env.reward()
            _ = obs
            _ = reward
            step_count += 1

    var elapsed = perf_counter_ns() - start
    var elapsed_ms = Float64(elapsed) / 1_000_000.0
    print("Episode rollout (", NUM_EPISODES, " episodes):", elapsed_ms, "ms (", elapsed_ms / NUM_EPISODES, "ms/episode)")


fn benchmark_combined_operations():
    print("\n--- Combined Operations Benchmark (Optimized) ---")

    # Simulate what happens in a typical RL step
    var iterations = NUM_ITERATIONS
    var start = perf_counter_ns()

    for _ in range(iterations):
        var env = PuzzleEnv(3, 3, 5, 2, 20)
        env.reset()
        var obs = env.observe()
        var masks = env.masks()
        var action = 0
        for i in range(NUM_ACTIONS):
            if masks[i]:
                action = i
                break
        env.step(action)
        var reward = env.reward()
        var is_final = env.is_final()
        _ = obs
        _ = reward
        _ = is_final

    var elapsed = perf_counter_ns() - start
    var elapsed_us = Float64(elapsed) / 1000.0
    print("Combined RL step (", iterations, " iterations):", elapsed_us / 1000.0, "ms (", elapsed_us / iterations, "us/iter)")


fn main():
    print("============================================================")
    print("TwisterL Mojo Benchmark (Optimized with InlineArray)")
    print("============================================================")
    print("Optimizations applied:")
    print("  - InlineArray[Int, 9] for puzzle state (stack vs heap)")
    print("  - InlineArray[Bool, 4] for action masks")
    print("  - @always_inline for hot methods")
    print("  - Pre-allocated capacity for List operations")

    benchmark_env_operations()
    benchmark_episode_rollout()
    benchmark_combined_operations()

    print("\n============================================================")
    print("Benchmark complete!")
    print("============================================================")
