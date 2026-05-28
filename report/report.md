# BrainBlock Packing with Deep Reinforcement Learning

## Current Draft Status

This report is a working draft. It records the environment design, baseline results, DQN experiments, and the current reward-design conclusions. Final plots, sample solved boards, and polished discussion still need to be added.

## Problem Overview

BrainBlock is an 8 by 5 tetromino packing task. Each episode starts with a shuffled queue containing two copies each of `I`, `O`, `L`, `Z`, and `T`. At each step, the agent must place only the current queue-head piece. A successful episode places all 10 tetrominoes and covers all 40 board cells.

The task is difficult for reinforcement learning because locally legal moves can create later dead ends. A policy that only learns to place many pieces can still fail to solve the board.

## MDP Definition

The state is a Gymnasium dictionary containing:

- `board`: a 5 by 8 integer grid using `0` for empty cells and `1` through `5` for placed piece types.
- `current_piece`: scalar piece index from `0` to `4`, or `-1` when no piece remains.
- `remaining_counts`: length-5 vector counting remaining `I`, `O`, `L`, `Z`, and `T` pieces.

The action space is `Discrete(320)`, representing:

```text
orientation in {0..7} x x_anchor in {0..7} x y_anchor in {0..4}
```

The flattened action rule is:

```text
action_id = orientation * 40 + y * 8 + x
```

An action is legal only if every occupied cell of the oriented piece is inside the board and does not overlap an already-filled cell. The episode terminates when all pieces are placed or when an invalid action is selected.

## Environment and Baselines

The environment, legality checks, action encoding, observation encoding, rendering, and rollout metrics are implemented. The test suite currently passes in the project environment with 40 tests.

Two baseline policies were implemented:

- `random`: samples from all 320 actions.
- `legal_random`: samples uniformly from currently legal actions.

Earlier 500-episode evaluation showed that `legal_random` covered about 27.86 cells on average with a success rate around 0.002. This established that local legal placement is not sufficient for solving.

## DQN Implementation

The DQN agent uses:

- one-hot board encoding,
- one-hot current-piece encoding,
- normalized remaining-count features,
- a multilayer perceptron producing 320 Q-values,
- legal-action masking for exploration and greedy evaluation,
- replay buffer,
- target network,
- epsilon-greedy exploration,
- checkpointing and CSV logging.

The first smoke run confirmed that the training loop, checkpoint saving, and evaluation pipeline work end to end.

## Reward Experiments So Far

The first main experiment trained two rewards for five seeds each, using 3000 training episodes and 500 evaluation episodes per seed.

### Sparse Reward

```text
legal placement: +0.1
completion: +10.0
invalid action: -5.0
```

Sparse DQN improved average coverage over legal-random but did not become a reliable solver.

Mean over 5 seeds:

```text
success rate: 0.0008
mean covered area: 29.26 / 40
invalid-action rate: 0.120
```

The best evaluated sparse checkpoint so far was `sparse/seed_0/checkpoints/best.pt`:

```text
success rate: 0.002
mean episodic return: -4.214
mean episode length: 8.562
invalid-action rate: 0.117
mean final covered area: 30.256 / 40
```

### Shaped Reward

```text
legal placement: +0.5
completion bonus: +10.0
invalid action: -5.0
bad empty region: -1.0 per empty component whose size is not divisible by 4
no next legal action: -0.5
```

This reward performed worse than sparse. The penalty appears too strong or too noisy, causing the agent to underperform even the legal-random baseline.

Mean over 5 seeds:

```text
success rate: 0.0000
mean covered area: 26.00 / 40
invalid-action rate: 0.133
```

## Interpretation

The initial DQN learned partial packing but not full solving. It discovered behavior that places more pieces than legal-random on average, but success remained near zero. This suggests the agent is optimizing for survival or local packing quality rather than finding complete solution trajectories.

The likely issues are:

- legal-placement rewards encourage partial progress,
- full solutions are rare under epsilon-greedy exploration,
- plain DQN with an MLP struggles with long-horizon combinatorial planning,
- randomized piece queues increase the difficulty because the agent must generalize across many orderings.

## Second Reward Revision

The next experiment removed positive rewards for ordinary legal placements. This was motivated by the observation that sparse and shaped rewards still rewarded partial packing, so the agent learned to place many pieces without reliably solving the board.

### Terminal Reward

```text
legal non-terminal placement: 0.0
completion: +20.0
invalid action: -5.0
```

This tests whether removing partial reward prevents the agent from overvaluing incomplete packings.

### Shaped V2 Reward

```text
legal non-terminal placement: 0.0
completion: +20.0
invalid action: -3.0
bad empty region after at least 20 covered cells: -0.25 each
no next legal action after a legal move: -1.0
```

This keeps the reward mostly terminal, but adds mild late-game topology guidance. Unlike the first shaped reward, it avoids rewarding placement itself and delays the empty-region penalty until the board is at least half full.

## Second Reward Experiment Results

The second reward experiment trained `terminal` and `shaped_v2` for five seeds each, again using 3000 training episodes and 500 evaluation episodes per seed. These results were compared against the earlier `sparse` and `shaped` runs.

Mean over 5 seeds:

```text
reward      success rate    mean covered area    invalid-action rate
shaped      0.0000          26.00 / 40           0.133
shaped_v2   0.0080          24.79 / 40           0.138
sparse      0.0008          29.26 / 40           0.120
terminal    0.2000          31.11 / 40           0.097
```

The most important result was `terminal/seed_2`, which learned a deterministic policy that solved every evaluation episode:

```text
Evaluation with 500 episodes, seed 1000:
success rate: 1.000
mean episodic return: 20.000
mean episode length: 10.000
invalid-action rate: 0.000
mean final covered area: 40.000 / 40
```

This checkpoint was then evaluated more strongly on 1000 episodes with a different evaluation seed:

```text
Evaluation with 1000 episodes, seed 2000:
success rate: 1.000
mean episodic return: 20.000
mean episode length: 10.000
invalid-action rate: 0.000
mean final covered area: 40.000 / 40
```

This shows that the policy is not merely solving one fixed piece order. It generalizes across a large set of shuffled queues.

## Reward Comparison Discussion

The reward comparison supports a clear conclusion: positive rewards for ordinary legal placement were misleading. They encouraged the agent to optimize partial board coverage rather than full completion. The pure terminal reward produced the best final policy because it aligned the learning signal directly with the true objective: solve the board or fail.

However, the terminal reward is still seed-sensitive. Among five terminal training seeds, one produced a perfect evaluated solver, while the other four failed to solve reliably. This suggests that terminal reward can work, but exploration remains a bottleneck. Some training runs discover successful trajectories early enough for DQN to reinforce them; other runs remain stuck in partial-packing behavior.

`shaped_v2` produced occasional successful episodes, but it did not outperform the pure terminal reward. Its topology penalties may still interfere with exploration, or they may not provide enough useful guidance for this board geometry.

Based on these results, the selected reward for the current best model is `terminal`.

## Next Experiments

The main remaining experimental questions are:

- Can the terminal reward be made more reliable across training seeds?
- Does longer training or slower epsilon decay increase the probability of finding the successful terminal policy?
- Can successful rollouts from `terminal/seed_2` be used as demonstrations or replay-buffer seeds?
- Does evaluating `best.pt` rather than `latest.pt` improve non-winning seeds?

## Remaining Work

- Generate learning curves from training logs.
- Save and visualize representative successful and failed rollouts.
- Add at least five solved boards from `terminal/seed_2`.
- Consider curriculum learning or search-generated demonstration data to improve seed-to-seed reliability.
- Prepare final report PDF and presentation demo.

## AI and Tool Usage

AI assistance was used for planning, code generation, debugging, and report drafting. The group remains responsible for reviewing, validating, and explaining all submitted code and experimental conclusions.
