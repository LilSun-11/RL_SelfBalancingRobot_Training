# 4. Import the Robot into Isaac Lab

Isaac Lab describes a robot with an `ArticulationCfg`: where its model comes from, how the physics
engine should treat it, where it starts, and which **actuators** drive its joints. In this chapter
you write that configuration for the URDF from chapter 3.

!!! abstract "Learning objectives"
    - Load a URDF directly with `UrdfFileCfg`.
    - Choose the physics settings for a small, light robot.
    - Model the DC motors as torque-controlled actuators with a reaction delay.

You will create **one new file** in the task folder:

```text
source/SelfBalancing/SelfBalancing/tasks/manager_based/selfbalancing/robot.py   NEW
```

```bash
cd ~/SelfBalancing
gedit source/SelfBalancing/SelfBalancing/tasks/manager_based/selfbalancing/robot.py
```

The sections below build the file step by step; [4.5](#45-the-complete-file) gives the complete
file to paste.

## 4.1 Point Isaac Lab at the URDF

`robot.py` lives 6 folders below the project root
(`source/SelfBalancing/SelfBalancing/tasks/manager_based/selfbalancing/`). Going up 6 folders gives
`~/SelfBalancing`, so the URDF path works wherever the project is:

```python
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 6)))
ROBOT_URDF_PATH = os.path.join(_PROJECT_ROOT, "assets", "RobotTwoWheel", "urdf", "SelfBalancingRobot_simplified.urdf")
```

## 4.2 Spawn settings

```python
TwoWheel_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=ROBOT_URDF_PATH,
        fix_base=False,                 # the robot moves freely (it is not bolted to the world)
        root_link_name="base_link",
        merge_fixed_joints=True,        # upper plate, battery and motors become part of base_link
        self_collision=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",         # no position/velocity target: the policy commands torque
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
```

| Setting | Why |
|---|---|
| `UrdfFileCfg` | Converts the URDF when the simulation starts, so the simulated robot always matches the file. |
| `merge_fixed_joints=True` | Leaves only 3 rigid bodies (`base_link` ≈ 1.09 kg, `wheel1`, `wheel2`). Fewer bodies make the simulation faster and more stable, and the merged body keeps the combined mass and inertia. |
| `target_type="none"` | The wheels are not driven to a position or speed; the actuator below applies the torque the policy asks for. |
| `enable_gyroscopic_forces=True` | Spinning wheels produce gyroscopic effects that matter when the robot turns. |

## 4.3 Initial state

```python
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.070),          # 0.068 m ground clearance (chapter 3.6) + 2 mm margin
        joint_pos={"wheel1_motor1_joint": 0.0, "wheel2_motor2_joint": 0.0},
    ),
```

The task in chapter 5 adds a random tilt to this pose at every reset.

## 4.4 Model the motors

This is the part that matters most for sim-to-real. The robot's DC motors are driven by PWM through
an H-bridge. The policy outputs a value in [−1, 1] per wheel that becomes a torque, and the real
motor responds to a change of direction with a noticeable lag.

```python
    actuators={
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=["wheel1_motor1_joint", "wheel2_motor2_joint"],
            effort_limit=0.49,          # Nm, maximum wheel torque
            stiffness=0.0,              # no position control
            damping=0.002,              # small velocity-dependent loss (like back-EMF)
            min_delay=2,                # delay of 2..8 physics steps = 10..40 ms,
            max_delay=8,                # resampled at every reset
        ),
    },
)
```

- With `stiffness=0` a PD actuator is a pure **torque** actuator: the applied torque is the
  commanded effort minus `damping × wheel velocity`, clipped to `effort_limit`.
- `DelayedPDActuatorCfg` applies each command a random number of physics steps late. A policy
  trained against an instant motor learns to react at the last moment and falls on the real robot;
  one trained against a range of delays learns to act early enough for all of them.

!!! note "Measure your own motor"
    `effort_limit=0.49` Nm and the 10–40 ms delay are estimates for the JGB37-520 motor. If you build
    a different robot, use your motor's stall torque (datasheet) and the time it takes to reverse
    direction.

## 4.5 The complete file

Paste this into `robot.py` and save:

??? example "`source/SelfBalancing/SelfBalancing/tasks/manager_based/selfbalancing/robot.py` (complete file)"

    ```python
    --8<-- "docs/code/source/SelfBalancing/SelfBalancing/tasks/manager_based/selfbalancing/robot.py"
    ```

[:material-download: Download the file](code/source/SelfBalancing/SelfBalancing/tasks/manager_based/selfbalancing/robot.py){ .md-button download }

`robot.py` only defines `TwoWheel_CFG`; nothing uses it yet. Chapter 5 puts it into the task's
scene, and chapter 5.11 checks that the robot spawns correctly.

!!! success "Checkpoint"
    `robot.py` is in the task folder and defines `TwoWheel_CFG`.
