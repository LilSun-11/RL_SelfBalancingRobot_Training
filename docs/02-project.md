# 2. Create the Project

Isaac Lab ships a template generator that creates a complete, installable project for a new robot
task: a Python package, the train/play scripts and a placeholder task. In this chapter you generate
that project. In the following chapters you replace the placeholder with the self-balancing robot,
one file at a time.

!!! abstract "Learning objectives"
    - Generate an external Isaac Lab project with the template generator.
    - Install it and run the placeholder task.
    - Know which files you will add or change in the rest of the tutorial.

## 2.1 Run the template generator

Open a new terminal, go to the Isaac Lab folder and activate the environment:

```bash
cd ~/IsaacLab
conda activate env_isaaclab
```

Run the Isaac Lab script with the `--new` argument:

```bash
./isaaclab.sh --new
```

The generator asks a few questions. Use the arrow keys to move, the **space bar** to select an
option in a list, and **Enter** to confirm:

| Question | Answer |
|---|---|
| **Task type** | `External` |
| **Project path** | Press Enter to keep the default (your home folder, e.g. `/home/<user>/`) |
| **Project name** | `SelfBalancing` |
| **Isaac Lab workflow** | `Manager-based | single-agent` |
| **RL library** | `rsl_rl` |
| **RL algorithms** (if asked) | `PPO` |

When it finishes it prints:

```text
Project 'SelfBalancing' generated successfully in /home/<user>/ path.
```

!!! note "Names generated from `SelfBalancing`"
    The generator derives the other names from the project name. You will see them throughout the
    tutorial:

    | Item | Name |
    |---|---|
    | Project folder | `~/SelfBalancing` |
    | Python package | `SelfBalancing` |
    | Task folder | `selfbalancing` |
    | Task id | `Template-Selfbalancing-v0` |
    | Environment config class | `SelfbalancingEnvCfg` |

## 2.2 Install the project

Install the package in editable mode, so your changes take effect without reinstalling:

```bash
cd ~/SelfBalancing
python -m pip install -e source/SelfBalancing
```

All commands in the rest of the tutorial are run from `~/SelfBalancing` unless a `cd` says
otherwise.

## 2.3 Look at what was generated

```text
~/SelfBalancing/
├── scripts/
│   ├── list_envs.py                ← lists the registered tasks
│   ├── zero_agent.py               ← runs a task with zero actions
│   ├── random_agent.py             ← runs a task with random actions
│   └── rsl_rl/
│       ├── train.py                ← trains a policy (chapter 6)
│       ├── play.py                 ← plays and exports a policy (chapter 7)
│       └── cli_args.py
└── source/SelfBalancing/
    ├── setup.py
    └── SelfBalancing/
        └── tasks/manager_based/selfbalancing/      ← the task folder
            ├── __init__.py                         ← registers the task id
            ├── selfbalancing_env_cfg.py            ← the task definition (a cart-pole for now)
            ├── agents/
            │   └── rsl_rl_ppo_cfg.py               ← PPO settings
            └── mdp/
                ├── __init__.py
                └── rewards.py                      ← custom reward functions
```

The path `source/SelfBalancing/SelfBalancing/tasks/manager_based/selfbalancing/` appears in almost
every chapter. This tutorial calls it **the task folder**.

The placeholder task balances a cart-pole. It already works. List it and train it for a few
iterations to check the installation:

```bash
python scripts/list_envs.py
python scripts/rsl_rl/train.py --task=Template-Selfbalancing-v0 --headless --max_iterations 20
```

`list_envs.py` prints a table with `Template-Selfbalancing-v0`. The training prints 20 iterations
and saves checkpoints in `logs/rsl_rl/cartpole_direct/`. You can delete that `logs/` folder
afterwards.

## 2.4 Your roadmap

By the end of chapter 5 the project contains these files. **New** files are created by you;
**changed** files were generated and you replace or edit them.

```text
~/SelfBalancing/
├── assets/RobotTwoWheel/urdf/
│   └── SelfBalancingRobot_simplified.urdf   NEW      chapter 3
└── source/SelfBalancing/SelfBalancing/tasks/manager_based/selfbalancing/
    ├── __init__.py                          CHANGED  chapter 5.9
    ├── robot.py                             NEW      chapter 4
    ├── selfbalancing_env_cfg.py             CHANGED  chapter 5.8 (replaced)
    ├── agents/
    │   └── rsl_rl_ppo_cfg.py                CHANGED  chapter 5.10
    └── mdp/
        ├── __init__.py                      CHANGED  chapter 5.2 (replaced)
        ├── observations.py                  NEW      chapter 5.3
        ├── commands.py                      NEW      chapter 5.4
        ├── events.py                        NEW      chapter 5.5
        ├── rewards.py                       CHANGED  chapter 5.6 (replaced)
        └── curriculums.py                   NEW      chapter 5.7
```

Chapter 8 then adds `tools/export_policy_header.py` and a `firmware/` folder for the robot.

!!! info "Reference solution"
    The finished project, plus the firmware and the trained robot's demo, is on GitHub:
    [LilSun-11/RL_SelfBalancingRobot_Training](https://github.com/LilSun-11/RL_SelfBalancingRobot_Training).
    Use it to compare when something does not work, after trying it yourself.

!!! success "Checkpoint"
    `~/SelfBalancing` exists, `list_envs.py` shows `Template-Selfbalancing-v0`, and the cart-pole
    placeholder trains.
