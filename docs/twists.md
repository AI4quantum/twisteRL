# Twists (Permutation Symmetries) in twisteRL

Twists are twisteRL's way to describe exact permutation symmetries that exist inside an environment.
Instead of training a policy to rediscover that symmetries exist (for example, that swapping qubits
across a symmetric coupling map produces an equivalent observation/action space), the environment
hands the policy explicit permutations that it can use for data augmentation or symmetry-aware heads.
By repeatedly doing and undoing these permutations you also reduce the chance of deadlocks and gain a 
lightweight form of regularization because the agent sees equivalent states under many orderings.

## Where Twists Are Used
- Every environment implements the `twisterl::rl::env::Env` trait. The trait includes a `twists`
  method that returns `(Vec<Vec<usize>>, Vec<Vec<usize>>)` representing valid permutations on the
  flattened observation array and matching permutations on the discrete action space
  (`rust/src/rl/env.rs:33`).
- When an environment is instantiated from Python via `prepare_algorithm`, twisteRL immediately calls
  `env.twists()` and forwards the returned permutations to the policy constructor
  (`src/twisterl/utils.py:120`). The policy can then symmetrize logits, average values, or augment
  rollouts without extra environment queries.

## Data Contract
1. **Observation permutations (`obs_perms`)** are expressed in the same flattened index space
   produced by the environment’s `observe()` method. Each permutation covers every index exactly once.
2. **Action permutations (`act_perms`)** must use the same ordering as `obs_perms`. TwisteRL
   assumes `act_perms[i]` describes how to remap actions when `obs_perms[i]` is applied.
3. The length of the two permutation lists must match (`len(obs_perms) == len(act_perms)`), and the
   first permutation should usually be the identity so policies have a canonical ordering to fall back to.

## Implementing Twists in Rust Environments
1. **Compute permutations once** when the environment is constructed. Store the resulting vectors on
   the struct so you can reuse them without recomputing each step.
2. **Return cached permutations** from the `twists` method by cloning or otherwise referencing the
   stored vectors. This keeps the call cheap even when policies request twists frequently.
3. **Gate toggles through config**. Consider exposing a `use_perms` or `add_perms` flag so users can
   disable symmetries if they want to benchmark raw performance or compare against non-symmetric runs.

### Tips for new envs
- If your observation is multi-dimensional, decide on a consistent flattening order and reuse it in
  `observe()`, `obs_shape()`, and permutation computation.
- Keep permutations short: only add a symmetry when it actually preserves the transition dynamics;
  incorrect permutations can break training stability.
- Store permutations on the struct instead of recomputing them each `twists()` call to avoid extra
  allocations during training.

## Implementing Twists in Python Environments
Python environments exposed through `PyEnv` can mirror the same pattern:

1. **Detect graph/device symmetries** using domain-specific tooling. Capture any permutation that
   leaves the transition structure unchanged.
2. **Sample a permutation for every observation** if you want trajectories to naturally explore each
   orbit; this mimics the way many structured environments randomize qubit or tile order.
3. **Expose action permutations** through the PyO3 wrapper so the policy receives matching
   permutations. When porting a Python env to Rust, copy the action/observation permutation lists into
   the Rust struct and return them from `twists()`.

## Verifying Your Twists
1. Call `env.twists()` from Python and check that each permutation is a rearrangement of
   `range(len(observe()))` and `range(num_actions())`.
2. Run a short training job with and without permutations enabled. If permutations are correct you
   should see either faster convergence or identical performance; regressions usually mean the
   action-and-observation permutations are misaligned.
3. For debugging, temporarily limit the permutation list to `[identity]` and re-enable additional
   symmetries one at a time.

By explicitly documenting and exposing twists, twisteRL policies gain symmetry awareness for free,
leading to higher data efficiency on structured problems such as puzzle solvers and quantum circuit
optimization.
