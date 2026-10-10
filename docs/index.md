# Train and Deploy a Self-Balancing Robot with Isaac Lab

**Physical AI Lab (PAI Lab), University of Ulsan** · Sim-to-Real Course

In this tutorial you take a small two-wheeled robot from a set of measurements to a real robot that
balances by itself, driven by a neural network you trained in simulation. Along the way you build
the robot model, write a reinforcement learning (RL) task in Isaac Lab, train it with PPO, check the
result in simulation, and run the network on an ESP32 microcontroller at 100 Hz.

| Simulation (Isaac Lab, trained policy) | Real robot (ESP32) |
|---|---|
| [![Simulation demo](media/RL_SBR.gif)](media/RL_SBR.webm) | [![Real robot demo](media/Real_SBR.gif)](media/Real_SBR.mp4) |

## What you will build

The robot is a **TWIP** (two-wheeled inverted pendulum): a chassis on two independently driven
wheels, like a small Segway. It cannot stand on its own. The controller has to keep moving the
wheels under the center of mass, 100 times per second.

The trained policy is a small neural network (MLP, 10 inputs → 32 → 32 → 2 outputs). It reads the
tilt and rotation from an IMU, the wheel speeds from the encoders, and two commands (forward speed
and heading), and outputs a torque for each wheel. The robot can balance in place, drive forward
and backward, and turn to a commanded heading.

## The workflow

![End-to-end workflow: physical robot -> URDF model -> Isaac Sim -> training in Isaac Lab (PPO) -> policy evaluation -> export .onnx to a C header -> deploy to the robot](media/workflow.png)

| Chapter | You will | Result |
|---|---|---|
| [1. Set up your machine](01-setup.md) | Install Isaac Sim, Isaac Lab, this project and PlatformIO | A working `env_isaaclab` environment |
| [2. Build the robot URDF](02-urdf.md) | Describe the robot's links, joints, masses and inertias | `SelfBalancingRobot_simplified.urdf` |
| [3. Import the robot into Isaac Lab](03-import.md) | Load the URDF and model the motors | `robot.py` (`TwoWheel_CFG`) |
| [4. Create the RL task](04-task.md) | Define observations, actions, commands, rewards, randomization | `selfbalancing_env_cfg.py` |
| [5. Train the policy](05-train.md) | Run PPO on thousands of robots in parallel | Checkpoints and TensorBoard curves |
| [6. Evaluate and export](06-evaluate.md) | Watch the policy, pick a checkpoint, export it | `policy.onnx` |
| [7. Sim-to-real deployment](07-sim-to-real.md) | Wire the robot, convert the network to C, flash the ESP32 | A balancing real robot |
| [8. Troubleshooting](08-troubleshooting.md) | Fix common problems | — |

## Before you start

You should be comfortable with:

- Basic Linux terminal use (`cd`, `ls`, editing files).
- Basic Python (classes, functions, reading someone else's code).
- Basic C/C++ and Arduino (`setup()` / `loop()`), for chapter 7.

You do **not** need prior experience with Isaac Lab or reinforcement learning; each chapter explains
the concepts it uses.

You need:

- A PC with **Ubuntu 22.04 or 24.04** and an **NVIDIA RTX GPU** (tested on an RTX 3080, 10 GB).
- For chapter 7: the robot hardware listed in [7.2](07-sim-to-real.md#72-hardware).

!!! info "Source code"
    Everything in this tutorial is in the repository
    [LilSun-11/RL_SelfBalancingRobot_Training](https://github.com/LilSun-11/RL_SelfBalancingRobot_Training).
    File paths in the chapters are relative to the repository root.

## Acknowledgements

Physical AI Lab (PAI Lab), **University of Ulsan**. Sim-to-Real Course, Hardware Team.
Supervisor: Prof. **Ahn Kyoung Kwan**.
