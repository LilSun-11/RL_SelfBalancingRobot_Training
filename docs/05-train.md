# 5. Train the Policy

With the task defined, training is one command. In this chapter you start a run, learn to read its
progress, and learn what to change when a run does not converge.

!!! abstract "Learning objectives"
    - Run PPO training headless on thousands of parallel robots.
    - Read the training curves in TensorBoard.
    - Resume a run and change settings from the command line.
    - Recognize the common failure modes of this task.

## 5.1 The PPO configuration

[`agents/rsl_rl_ppo_cfg.py`](https://github.com/LilSun-11/RL_SelfBalancingRobot_Training/blob/master/source/SelfBalancing/SelfBalancing/tasks/manager_based/selfbalancing/agents/rsl_rl_ppo_cfg.py)
sets the network and the algorithm:

```python
@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32          # steps collected per robot per iteration
    max_iterations = 1000
    save_interval = 50              # save a checkpoint every 50 iterations
    experiment_name = "selfbalancing"
    clip_actions = 1.0              # actions are clipped to [-1, 1] before reaching the env
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[32, 32],  # small enough to run on an ESP32
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=1.0e-3, schedule="adaptive", desired_kl=0.01,
        gamma=0.99, lam=0.95, clip_param=0.2, entropy_coef=0.005,
        num_learning_epochs=5, num_mini_batches=4,
        value_loss_coef=1.0, use_clipped_value_loss=True, max_grad_norm=1.0,
    )
```

With 8196 robots × 32 steps, one iteration collects about 262 000 transitions (≈ 44 minutes of robot
experience at 100 Hz).

!!! note "Two choices made for deployment"
    - **`actor_hidden_dims=[32, 32]`**: the network must run in well under 10 ms on an ESP32. This
      one has 1474 parameters and runs in tens of microseconds.
    - **`actor_obs_normalization=False`**: no running mean/std to carry over to the firmware; the
      observations are already in small, similar ranges.

## 5.2 Start training

```bash
cd ~/RL_SelfBalancingRobot_Training
python scripts/rsl_rl/train.py --task=Template-SelfBalancing-v0 --headless
```

Useful flags:

| Flag | Effect |
|---|---|
| `--headless` | No viewport (much faster). Remove it to watch the robots learn. |
| `--num_envs <N>` | Number of parallel robots (default 8196). Lower it if the GPU runs out of memory. |
| `--max_iterations <N>` | Number of PPO iterations. |
| `--seed <N>` | Random seed. |
| `--run_name <name>` | Suffix for the log folder, to tell runs apart. |

Logs and checkpoints are written to `logs/rsl_rl/selfbalancing/<date>_<time>[_<run_name>]/`
(`model_<iteration>.pt`, the TensorBoard event file, and `params/` with the exact config used).

Training the full task (balance, velocity and heading) takes a few thousand iterations. Remember the
curriculum: the heading targets only start after 2000 iterations.

## 5.3 Change settings without editing files

Any field of the environment or agent config can be overridden with Hydra syntax after the normal
flags:

```bash
python scripts/rsl_rl/train.py --task=Template-SelfBalancing-v0 --headless \
    env.rewards.action_rate.weight=-0.2 \
    env.commands.target_velocity.ranges="[-0.1,0.1]" \
    env.curriculum.yaw_range=null \
    agent.max_iterations=3000
```

`null` disables a term. This is the quickest way to try a change before committing it to the config.

## 5.4 Watch the curves in TensorBoard

In a second terminal:

```bash
conda activate env_isaaclab
cd ~/RL_SelfBalancingRobot_Training
tensorboard --logdir logs/rsl_rl/selfbalancing --port 6006
```

Open <http://localhost:6006>. The most useful curves:

| Curve | What a healthy run shows |
|---|---|
| `Train/mean_episode_length` | Rises toward 2000 steps (20 s): the robots stop falling. |
| `Episode_Termination/fell_over` | Drops toward 0. |
| `Episode_Termination/time_out` | Rises toward 1. |
| `Train/mean_reward` | Rises, then flattens. |
| `Metrics/target_velocity/error_vel` | Mean speed error (m/s), going down. |
| `Metrics/target_yaw/error_yaw` | Mean heading error (rad); rises when the curriculum enables heading targets, then goes down. |
| `Episode_Reward/<term>` | Contribution of each reward term; shows which one dominates. |
| `Loss/value_function`, `Policy/mean_noise_std` | Value loss should stay finite; the action noise should slowly decrease. |

## 5.5 Resume a run

```bash
python scripts/rsl_rl/train.py --task=Template-SelfBalancing-v0 --headless \
    --resume --load_run <run_folder> --checkpoint model_<N>.pt --max_iterations <more>
```

- `--max_iterations` adds that many iterations on top of the checkpoint.
- You can change reward weights, randomization or commands between runs, but **not** the number of
  observations or actions (the network's input and output size).
- The curriculum counts steps from the start of each `train.py` process. When resuming a run that is
  already past 2000 iterations, add `env.curriculum.yaw_range=null` to keep the full heading range.

## 5.6 When training goes wrong

These are problems met while developing this project, and what fixed them.

??? failure "Reward suddenly drops to −10¹⁸, then `normal expects all elements of std >= 0.0`"
    The previous actions are fed back as observations. Without clipping, the policy can output huge
    actions (the torque saturates anyway, so there is no physical cost), which feed back and grow
    until the `action_rate` penalty overflows. **Fix:** `clip_actions = 1.0` in the agent config.

??? failure "A tracking penalty is hundreds per step and the run never converges"
    A squared penalty with a small `norm_scale` explodes after a push or a fall (e.g. an error of
    0.5 m/s with `norm_scale=0.05` costs 100 × the weight per step). **Fix:** keep the squared penalty
    loose (`norm_scale` 0.2 m/s, 1.0 rad) and get precision from the bounded exponential bonus.

??? failure "The robot balances, but the wheels buzz back and forth (bang-bang control)"
    The policy switches between full forward and full backward torque every step. **Fix:** increase
    the `action_rate` penalty, and check that no reward term is so narrow that only chattering can
    satisfy it (bonus `std` too small).

??? failure "The robot ignores the command and just stands still"
    The bonus `std` is so wide that ignoring the command still earns most of it, or the tracking
    weights are small next to `alive` and `upright`. Compare the `Episode_Reward/*` curves to see
    which term dominates.

??? failure "The policy works in sim but spins in place on the real robot"
    In a perfectly symmetric simulation the two wheels never disagree, so the policy never learns to
    correct it. **Fix:** randomize each wheel's friction independently (`asymmetry` in
    `randomize_wheel_friction_motor`), observe the yaw rate, and command a heading so drift is an
    error.

!!! success "Checkpoint"
    `mean_episode_length` reaches ~2000 steps, `fell_over` is near 0, and the tracking errors go
    down after the curriculum switches on the heading targets.
