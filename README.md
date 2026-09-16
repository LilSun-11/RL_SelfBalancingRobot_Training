# Self-Balancing TWIP Robot (Isaac Lab)

## Overview

This project trains a two-wheeled self-balancing robot (a "TWIP" — two-wheeled inverted pendulum)
with reinforcement learning in Isaac Lab, then deploys the trained policy to real hardware.

The end-to-end workflow is:

```text
recreate the 2-wheel robot as a URDF -> import it into Isaac Sim
      -> train a balancing / velocity-tracking task with RSL-RL (PPO)
      -> evaluate the policy and export it to .onnx
      -> convert the exported policy to a C array/header -> flash it onto an ESP32
```

- The robot chassis was modeled in CAD and exported as a URDF with STL meshes
  (`assets/RobotTwoWheel/`), imported directly into Isaac Sim at simulation start (no prebaked USD),
  so the simulated robot never drifts out of sync with the CAD source.
- Training uses Isaac Lab's manager-based RL environment (`source/SelfBalancing/`) with RSL-RL/PPO:
  the robot balances upright while tracking a randomly commanded forward/backward body velocity.
- `scripts/rsl_rl/play.py` evaluates a trained checkpoint and automatically exports it to both
  `policy.pt` (JIT) and `policy.onnx` under `logs/rsl_rl/<experiment>/<run>/exported/`.
- Converting that `.onnx` file into a C array/header and flashing it onto an ESP32 is the final,
  sim-to-real step of the pipeline; this repository currently covers everything up to the ONNX
  export, and the ONNX-to-firmware conversion/flashing tooling is not included here yet.

This structure (and the requirements/troubleshooting sections below) follows the layout of
[sim2real-line-following-robot](https://github.com/SangHuynhVan272/sim2real-line-following-robot), a
similar sim-to-real Isaac Lab -> ESP32 project, adapted to this robot and to a Ubuntu-only workflow.

**Keywords:** self-balancing robot, TWIP, reinforcement learning, Isaac Lab, RSL-RL, PPO, sim-to-real, ESP32

## 1. What is included

| Path                                                             | Purpose                                                      |
| ----------------------------------------------------------------- | ------------------------------------------------------------ |
| `assets/RobotTwoWheel/`                                          | URDF + STL meshes for the physical robot chassis              |
| `source/SelfBalancing/SelfBalancing/tasks/manager_based/selfbalancing/` | Robot config, MDP terms (actions/commands/events/observations/rewards/terminations), env config |
| `source/SelfBalancing/SelfBalancing/tasks/manager_based/selfbalancing/agents/` | RSL-RL PPO hyperparameters                                     |
| `scripts/rsl_rl/train.py`, `scripts/rsl_rl/play.py`               | Train / evaluate + export (.pt and .onnx) programs             |
| `scripts/list_envs.py`, `scripts/zero_agent.py`, `scripts/random_agent.py` | Sanity-check scripts (list registered tasks, zero/random action rollouts) |
| `logs/rsl_rl/selfbalancing/<timestamp>/`                          | Per-run checkpoints, TensorBoard logs, and `exported/policy.onnx` |

## 2. Requirements

This guide covers Ubuntu only (22.04 LTS or 24.04 LTS).

- an NVIDIA RTX GPU with a recent driver (this project was developed and tested with the setup
  below); NVIDIA publishes the current GPU/VRAM/driver requirements for your Isaac Sim version in the
  [Isaac Sim requirements docs](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) —
  run `nvidia-smi` first and repair the NVIDIA driver before installing anything if it fails;
- a stable internet connection for the first Isaac Sim launch (extension/shader downloads).

Tested with:

| Component | Version |
|---|---:|
| OS | Ubuntu 24.04 LTS |
| GPU / driver | NVIDIA RTX 3080 (10 GB VRAM) / 580.173.02 |
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | v2.3.2 |
| PyTorch | 2.7.0, CUDA 12.8 build |
| RSL-RL | 3.1.2 |
| NumPy | 1.26.0 |
| ONNX | 1.20.1 |

These are the versions this repository was validated against, not a hard requirement — but a
different `rsl-rl-lib`/Isaac Lab combination can change the RL config API (see Troubleshooting).

## 3. Install on Ubuntu 22.04 or 24.04

### 3.1 Create the Python environment

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
python -m pip install --upgrade pip
```

The prompt must begin with `(env_isaaclab)` for every command below.

### 3.2 Install Isaac Sim and PyTorch

```bash
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
python -m pip install --upgrade --force-reinstall torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

Start Isaac Sim once, accept the EULA, and let it finish downloading extensions/building shader
caches (can take more than ten minutes on the first launch), then close it:

```bash
isaacsim
```

### 3.3 Install Isaac Lab

```bash
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2
./isaaclab.sh --install rsl_rl
```

### 3.4 Install this project

Clone or copy this repository outside of the `IsaacLab` directory, then, using the Python
interpreter that has Isaac Lab installed:

```bash
python -m pip install -e source/SelfBalancing
```

## 4. Verify the installation

List the registered tasks (`Template-SelfBalancing-v0` and `Template-SelfBalancing-Play-v0` should
appear — note the search pattern `"Template-"` in `scripts/list_envs.py`):

```bash
python scripts/list_envs.py
```

Run a zero-action or random-action rollout to confirm the environment itself is configured
correctly, before spending time on training:

```bash
python scripts/zero_agent.py --task=Template-SelfBalancing-v0
python scripts/random_agent.py --task=Template-SelfBalancing-v0
```

### Set up IDE (Optional)

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the
  `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the
  absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file `.python.env` in the `.vscode` directory,
containing the python paths to all the extensions provided by Isaac Sim and Omniverse — this helps
with indexing for intelligent code suggestions.

### Setup as Omniverse Extension (Optional)

An example UI extension loads upon enabling your extension, defined in
`source/SelfBalancing/SelfBalancing/ui_extension_example.py`.

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project.
    - If not already present, also add the path to Isaac Lab's extension directory (`IsaacLab/source`).
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## 5. Train and export your policy

Two task variants are registered: `Template-SelfBalancing-v0` (training, 8196 parallel envs) and
`Template-SelfBalancing-Play-v0` (playback/evaluation, 36 envs, no observation noise — see
`SelfBalancingEnvCfg_PLAY` in `selfbalancing_env_cfg.py`). The robot's task is to stay upright while
tracking a randomly commanded forward/backward body velocity (-0.4 to 0.4 m/s).

### Train

```bash
python scripts/rsl_rl/train.py --task=Template-SelfBalancing-v0 --headless
```

Drop `--headless` to watch training in the viewport (much slower). Useful flags:

- `--num_envs <N>` — override the number of parallel environments.
- `--max_iterations <N>` — number of PPO iterations to run.
- `--seed <N>` — environment seed.

Logs and checkpoints are saved under `logs/rsl_rl/<experiment_name>/<timestamp>/` (`experiment_name`
is set in `agents/rsl_rl_ppo_cfg.py`).

### Resume training

```bash
python scripts/rsl_rl/train.py --task=Template-SelfBalancing-v0 --headless \
    --resume --load_run <run_dir_name> --checkpoint <model_XXX.pt> --max_iterations <N>
```

`--max_iterations` adds `<N>` more iterations on top of the resumed checkpoint, not a total target.
Omit `--load_run`/`--checkpoint` to auto-pick the most recent run and its latest checkpoint.
Resuming only works if the observation/action dimensions haven't changed since that checkpoint was
trained (other config changes, e.g. reward weights, are fine to resume across).

### Evaluate a trained policy and export it

```bash
python scripts/rsl_rl/play.py --task=Template-SelfBalancing-Play-v0 \
    --load_run <run_dir_name> --checkpoint <model_XXX.pt>
```

This both plays the policy in the viewport (add `--real-time` to pace it to real time, or `--video`
to record instead of watching live) and exports it, writing `policy.pt` (JIT) and `policy.onnx` to
`logs/rsl_rl/selfbalancing/<run_dir_name>/exported/`. `policy.onnx` is the artifact the next
(not-yet-included) step converts into a C array/header to flash onto the ESP32.

To compare runs (reward curves, episode length, termination breakdown, etc.) before picking a
checkpoint to evaluate, point TensorBoard at the project's log directory:

```bash
tensorboard --logdir logs/rsl_rl/selfbalancing --port 6006
```

Then open `http://localhost:6006` in a browser. `--logdir` can also point at a single run
(`logs/rsl_rl/selfbalancing/<run_dir_name>`) to inspect just that run, and `--port` can be omitted to
use TensorBoard's default port (6006).

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `python` points outside `env_isaaclab`, or `isaacsim`/`isaaclab.sh` not found | Run `conda activate env_isaaclab` and retry |
| `ModuleNotFoundError: isaaclab` or `isaaclab_rl` | Isaac Lab isn't installed in this environment — rerun `./isaaclab.sh --install rsl_rl` from the `IsaacLab` directory |
| `torch.cuda.is_available()` is `False` | Repair the NVIDIA driver, then reinstall the cu128 PyTorch command from step 3.2 |
| `RslRlOnPolicyRunnerCfg`/checkpoint errors mentioning `actor`/`critic`/`policy` fields | A newer `rsl-rl-lib` restructured the RL config; `train.py`/`play.py` already call `handle_deprecated_rsl_rl_cfg`/`handle_deprecated_rsl_rl_checkpoint` when available to bridge this — make sure Isaac Lab is up to date, or pin `rsl-rl-lib` to the tested version (3.1.2) |
| Task not listed by `scripts/list_envs.py` | Confirm the package was installed with `python -m pip install -e source/SelfBalancing` and that the task id still starts with `Template-` (see `scripts/list_envs.py`'s search pattern) |
| First Isaac Sim launch appears frozen | Wait — the first launch downloads extensions/builds shader caches, which can take more than ten minutes |
| Pylance is missing indexing for part of the extensions | Add the path to your extension in `.vscode/settings.json` under `"python.analysis.extraPaths"` (see below) |
| Pylance crashes / runs out of memory | Too many files are indexed — comment out unused Omniverse packages under `"python.analysis.extraPaths"` in `.vscode/settings.json` (see below) |

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/SelfBalancing"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```
