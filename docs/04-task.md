# 4. Create the RL Task

The robot can now be simulated. To train a controller you also need a **task**: what the policy
observes, what it controls, what it is asked to do, how it is rewarded, and when an attempt ends.
Isaac Lab's *manager-based* environments describe each of these as a list of small, named terms in
one configuration file.

!!! abstract "Learning objectives"
    - Create an Isaac Lab extension project and register a task.
    - Write each part of a manager-based environment: scene, actions, observations, commands,
      events, rewards, terminations and curriculum.
    - Choose observations that the real robot can also measure.

The result is
[`selfbalancing_env_cfg.py`](https://github.com/LilSun-11/RL_SelfBalancingRobot_Training/blob/master/source/SelfBalancing/SelfBalancing/tasks/manager_based/selfbalancing/selfbalancing_env_cfg.py)
plus the custom terms in `mdp/`.

## 4.1 Create the project

Isaac Lab can generate an external project (a Python package that lives outside the Isaac Lab
repository):

```bash
cd ~/IsaacLab
./isaaclab.sh --new
```

Choose an **external** project, the **manager-based** workflow and the **RSL-RL** library. This
repository was created that way; its task lives in:

```text
source/SelfBalancing/SelfBalancing/tasks/manager_based/selfbalancing/
├── __init__.py                 ← registers the task with Gymnasium
├── robot.py                    ← TwoWheel_CFG (chapter 3)
├── selfbalancing_env_cfg.py    ← the task (this chapter)
├── agents/
│   └── rsl_rl_ppo_cfg.py       ← PPO hyperparameters (chapter 5)
└── mdp/                        ← custom terms used by the task
    ├── observations.py
    ├── commands.py
    ├── events.py
    ├── rewards.py
    ├── curriculums.py
    └── terminations.py
```

`mdp/__init__.py` re-exports Isaac Lab's built-in terms (`from isaaclab.envs.mdp import *`) together
with the custom ones, so the config can use both through `mdp.<name>`.

## 4.2 Register the task

`__init__.py` gives the task an id that the train and play scripts look up:

```python
import gymnasium as gym

from . import agents

gym.register(
    id="Template-SelfBalancing-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.selfbalancing_env_cfg:SelfBalancingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
```

A second id, `Template-SelfBalancing-Play-v0`, points to `SelfBalancingEnvCfg_PLAY` (fewer robots,
no sensor noise) for evaluation.

## 4.3 The environment at a glance

```python
@configclass
class SelfBalancingEnvCfg(ManagerBasedRLEnvCfg):
    scene: TwoWheelSceneCfg = TwoWheelSceneCfg(num_envs=8196, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumsCfg = CurriculumsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2              # policy runs every 2 physics steps
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 200            # 200 Hz physics -> 100 Hz control, same as the firmware
        self.sim.render_interval = self.decimation
```

!!! important "Match the control rate"
    `sim.dt × decimation = 0.01 s`, i.e. the policy acts at **100 Hz**. The firmware in chapter 7
    runs its loop at exactly the same rate. If the two rates differ, the policy's sense of time
    (velocities, delays, how fast to react) is wrong on the real robot.

The sections below go through each part.

## 4.4 Scene

A ground plane, a light, and the robot from chapter 3, copied once per environment
(`{ENV_REGEX_NS}` becomes `/World/envs/env_0`, `env_1`, …):

```python
@configclass
class TwoWheelSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5, dynamic_friction=1.2, restitution=0.0,
                friction_combine_mode="max", restitution_combine_mode="min",
            ),
        ),
    )
    robot: ArticulationCfg = TwoWheel_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
```

## 4.5 Actions

One torque command per wheel. The policy outputs values around [−1, 1]; `scale` turns them into Nm:

```python
WHEEL_JOINT_NAMES = ["wheel1_motor1_joint", "wheel2_motor2_joint"]  # left (+Y), right (-Y)
MOTOR_TORQUE_MAX = 0.49  # Nm

@configclass
class ActionsCfg:
    wheel_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=WHEEL_JOINT_NAMES, scale=MOTOR_TORQUE_MAX,
    )
```

Two independent actions (instead of one shared torque) let the policy steer, and correct the drift
caused by two motors that are never exactly alike.

## 4.6 Observations

!!! tip "Rule: only observe what the real robot can measure"
    The simulator knows everything (exact position, true velocity, contact forces…). The real robot
    only has an IMU and two wheel encoders. Every observation below can be computed on the ESP32;
    that is what makes deployment possible.

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        pitch_angle = ObsTerm(func=mdp.imu_pitch_angle, noise=Gnoise(mean=0.0, std=0.02))
        pitch_rate = ObsTerm(func=mdp.imu_pitch_rate, noise=Gnoise(mean=0.0, std=0.04))
        yaw_angle = ObsTerm(
            func=mdp.relative_yaw, params={"command_name": "target_yaw"}, noise=Gnoise(mean=0.0, std=0.02)
        )
        yaw_rate = ObsTerm(func=mdp.imu_yaw_rate, noise=Gnoise(mean=0.0, std=0.04))
        wheel1_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[WHEEL_JOINT_NAMES[0]])},
            noise=Gnoise(mean=0.0, std=0.05),
        )
        wheel2_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[WHEEL_JOINT_NAMES[1]])},
            noise=Gnoise(mean=0.0, std=0.05),
        )
        last_action1 = ObsTerm(func=mdp.last_action_index, params={"index": 0})
        last_action2 = ObsTerm(func=mdp.last_action_index, params={"index": 1})
        velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_velocity"})
        yaw_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_yaw"})

        def __post_init__(self) -> None:
            self.enable_corruption = True    # add the noise above (training only)
            self.concatenate_terms = True    # one vector, in the order written here

    policy: PolicyCfg = PolicyCfg()
```

| # | Observation | Unit | On the real robot |
|---|---|---|---|
| 1 | pitch angle (tilt forward/back) | rad | IMU, complementary filter |
| 2 | pitch rate | rad/s | gyro X |
| 3 | yaw angle, relative to the start heading | rad, [−π, π] | gyro Z, integrated |
| 4 | yaw rate | rad/s | gyro Z |
| 5–6 | left / right wheel velocity | rad/s | encoders |
| 7–8 | previous action, left / right | [−1, 1] | the firmware remembers it |
| 9 | target forward velocity | m/s | user command |
| 10 | target yaw | rad | user command |

The custom terms are short. The pitch comes from the gravity vector seen in the body frame, which is
what a filtered IMU estimate gives on the real robot:

```python
def imu_pitch_angle(env):
    pg = env.scene["robot"].data.projected_gravity_b        # [0, 0, -1] when upright
    return torch.atan2(-pg[:, 0], -pg[:, 2]).unsqueeze(1)

def imu_pitch_rate(env):
    return env.scene["robot"].data.root_ang_vel_b[:, 1].unsqueeze(1)

def imu_yaw_rate(env):
    return env.scene["robot"].data.root_ang_vel_b[:, 2].unsqueeze(1)

def last_action_index(env, index: int):
    return env.action_manager.action[:, index].unsqueeze(1)
```

Why each group is there:

- **Noise** (`Gnoise`): real sensors are never exact. Training with noise keeps the policy from
  relying on perfectly clean signals.
- **Previous actions**: the motor applies commands 10–40 ms late (chapter 3.4). Seeing what it asked
  for a moment ago lets the policy know which corrections are already "on their way".
- **Commands**: the same network can stand still, drive or turn, depending on the last two inputs.

!!! warning "Signs and order are a contract"
    The observation order above, their units and their signs must be reproduced exactly by the
    firmware. For example, tilting toward +X gives a *negative* pitch angle, and the simulator's
    pitch rate is the *negative* of d(pitch angle)/dt. Chapter 7 shows how the firmware matches this.

## 4.7 Commands

Commands are the "goal" given to the robot, resampled during an episode so the policy learns to
follow changing instructions. Both are custom terms in `mdp/commands.py`:

```python
@configclass
class CommandsCfg:
    target_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        ranges=(-0.15, 0.15),               # m/s
        resampling_time_range=(5.0, 7.0),   # s
        rel_standing_envs=0.15,             # 15% of the time: stand still
        debug_vis=True,
    )
    target_yaw = mdp.UniformYawCommandCfg(
        asset_name="robot",
        ranges=(-math.pi, math.pi),         # rad, relative to the spawn heading
        resampling_time_range=(5.0, 7.0),
        max_step=math.radians(60.0),        # each new target within ±60° of the previous one
        standing_command_name="target_velocity",
        debug_vis=True,
    )
```

- **Target velocity**: a forward/backward speed along the robot's own X axis. In about 15 % of the
  intervals it is exactly 0, so the policy also learns to stand still.
- **Target yaw**: a heading relative to the direction the robot faced when it spawned (that
  direction is yaw 0). Each new target is at most 60° away from the previous one, wrapped to
  [−π, π]. A robot told to stand still keeps the heading it has.

!!! question "Why a heading and not a turning speed?"
    A turning-speed command is natural for a joystick, but a heading command can also hold the robot
    pointing the same way: with a target yaw of 0, any drift is an error the policy corrects. This
    is what stops the real robot from slowly spinning when it should stand still.

## 4.8 Events (domain randomization)

Events change the simulation at startup, at every reset, or at intervals. They are the main tool for
**sim-to-real transfer**: if the policy works for a whole range of masses, frictions and delays, the
real robot is likely to be one of them.

| Event | When | What it does |
|---|---|---|
| `wheel_friction` | startup | Wheel–ground friction 1.5 / 1.2 (matches the ground) |
| `randomize_com` | startup | Shifts the chassis center of mass by ±5 mm in X and Z |
| `randomize_mass` | startup | Scales the chassis mass by 0.8–1.2 (inertia recomputed) |
| `randomize_wheel_friction_motor` | startup | Gearbox friction 0.010–0.014 Nm, with each wheel scaled independently by ±10 % so left and right differ like real motors |
| `reset_base` | reset | Random initial tilt ±0.52 rad (±30°) and tilt rate ±0.5 rad/s |
| `reset_wheels` | reset | Wheels start at rest |
| `push_robot` | every 4–6 s | Kicks the chassis forward/backward (force range set in the config; 0 disables it) |

The actuator delay of chapter 3.4 is randomization too: it is resampled at every reset.

## 4.9 Rewards

The reward tells PPO what "good" means. Each term is computed every step, multiplied by its weight,
and summed.

| Term | Weight | Function | Purpose |
|---|---|---|---|
| `alive` | +1.0 | 1 per step | Survive |
| `terminating` | −50000 | 1 when the robot falls | Falling is by far the worst outcome |
| `upright` | −1000 | tilt² (from projected gravity) | Stay vertical |
| `upright_bonus` | +0.5 | exp(−tilt² / 0.1²) | Extra reward for being very close to vertical |
| `pitch_rate` | −0.1 | roll/pitch rate² | No wobbling |
| `wheel_speed` | −0.00005 | wheel velocity² | Don't spin the wheels needlessly |
| `velocity_tracking` | −5.0 | ((v_cmd − v) / 0.2)² | Follow the target velocity |
| `velocity_tracking_bonus` | +7.5 | exp(−(v_cmd − v)² / 0.05²) | Reward close velocity tracking |
| `yaw_tracking` | −0.5 | (yaw error / 1.0)² | Turn to the target heading |
| `yaw_tracking_bonus` | +2.0 | exp(−yaw error² / 0.2²) | Reward close heading tracking |
| `yaw_rate` | −0.5 | yaw rate² | Turn smoothly, don't spin |
| `action_rate` | −0.1 | (action − previous action)² | Smooth motor commands, no jitter |

The yaw error is the shortest signed angle between target and current yaw (179° vs −179° is a 2°
error, not 358°).

!!! tip "Penalty + bonus pairs"
    Each tracking goal has two terms. The squared **penalty** is unbounded and gives a strong pull
    when the error is large. The exponential **bonus** is bounded (0 to its weight) and only pays out
    near the target; its `std` sets how close "close" is. A `std` that is too wide pays out even when
    the command is ignored; one that is too narrow makes the policy chatter the wheels to stay
    inside it.

## 4.10 Terminations

```python
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)          # 20 s episode finished
    fell_over = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.7},   # ~40° tilt
    )
```

`time_out=True` tells PPO that running out of time is not a failure (the robot was still fine), so
it does not get the `terminating` penalty.

## 4.11 Curriculum

Learning to balance, drive and turn at once is hard at the start. A curriculum makes the task easier
first: for the first 2000 PPO iterations the target yaw stays at 0 (just balance, drive, and hold
the heading), then the full heading range is enabled.

```python
@configclass
class CurriculumsCfg:
    yaw_range = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.target_yaw.ranges",
            "modify_fn": mdp.step_value_curriculum,
            "modify_params": {
                "start_value": (0.0, 0.0),
                "end_value": (-math.pi, math.pi),
                "num_steps": 2000 * 32,       # iterations × steps per iteration
            },
        },
    )
```

## 4.12 The play configuration

```python
class SelfBalancingEnvCfg_PLAY(SelfBalancingEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 360
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False   # no sensor noise
        self.curriculum.yaw_range = None                     # full heading range right away
```

!!! success "Checkpoint"
    `python scripts/zero_agent.py --task=Template-SelfBalancing-v0 --num_envs 16` runs and prints
    tables of active observation, reward, event and termination terms that match this chapter.
