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
| [1. Set up your machine](01-setup.md) | Install Isaac Sim, Isaac Lab and PlatformIO | A working `env_isaaclab` environment |
| [2. Create the project](02-project.md) | Generate an Isaac Lab project from the template | `~/SelfBalancing` |
| [3. Build the robot URDF](03-urdf.md) | Describe the robot's links, joints, masses and inertias | `SelfBalancingRobot_simplified.urdf` |
| [4. Import the robot into Isaac Lab](04-import.md) | Load the URDF and model the motors | `robot.py` |
| [5. Create the RL task](05-task.md) | Write observations, commands, events, rewards and the environment config | `mdp/*.py`, `selfbalancing_env_cfg.py` |
| [6. Train the policy](06-train.md) | Run PPO on thousands of robots in parallel | Checkpoints and TensorBoard curves |
| [7. Evaluate and export](07-evaluate.md) | Watch the policy, pick a checkpoint, export it | `policy.onnx` |
| [8. Sim-to-real deployment](08-sim-to-real.md) | Wire the robot, add the converter and firmware, flash the ESP32 | A balancing real robot |
| [9. Troubleshooting](09-troubleshooting.md) | Fix common problems | — |

You build the whole project yourself, file by file. Every file you create is shown in full on the
page where you create it.

## Before you start

You should be comfortable with:

- Basic Linux terminal use (`cd`, `ls`, editing files).
- Basic Python (classes, functions, reading someone else's code).
- Basic C/C++ and Arduino (`setup()` / `loop()`), for chapter 8.

You do **not** need prior experience with Isaac Lab or reinforcement learning; each chapter explains
the concepts it uses.

You need:

- A PC with **Ubuntu 22.04 or 24.04** and an **NVIDIA RTX GPU** (tested on an RTX 3080, 10 GB).
- For chapter 8: the robot hardware listed in [8.2](08-sim-to-real.md#82-hardware).

!!! info "Reference solution"
    The finished project is in the repository
    [LilSun-11/RL_SelfBalancingRobot_Training](https://github.com/LilSun-11/RL_SelfBalancingRobot_Training).
    Build your own first; use it to compare when something does not work.

## Acknowledgements

Physical AI Lab (PAI Lab), **University of Ulsan**. Sim-to-Real Course, Hardware Team.
Supervisor: Prof. **Ahn Kyoung Kwan**.
