# 1. Set Up Your Machine

In this chapter you install everything the rest of the tutorial needs: Isaac Sim (the physics
simulator), Isaac Lab (the robot-learning framework on top of it), this project, and PlatformIO
(to flash the robot's microcontroller later).

!!! abstract "Learning objectives"
    - Create the `env_isaaclab` Python environment.
    - Install Isaac Sim 5.1, Isaac Lab v2.3.2 and this project.
    - Install PlatformIO.
    - Confirm the training task is registered and runs.

Every command block can be copied and pasted into a terminal (`Ctrl + Alt + T`) as is.

## Tested versions

| Component | Version |
|---|---|
| OS | Ubuntu 24.04 LTS (22.04 also works) |
| GPU / driver | NVIDIA RTX 3080 (10 GB) / 580.173.02 |
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | v2.3.2 |
| PyTorch | 2.7.0 (CUDA 12.8) |
| RSL-RL | 3.1.2 |

Run `nvidia-smi` first. If it fails, repair the NVIDIA driver before installing anything else. The
current hardware and driver requirements are in the
[Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).

## 1.1 System packages and Miniconda

```bash
sudo apt update
sudo apt install -y git curl wget cmake build-essential python3-venv
```

Skip the next block if `conda --version` already works.

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## 1.2 Create the Python environment

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
python -m pip install --upgrade pip
```

!!! warning
    The prompt must begin with `(env_isaaclab)` for every command in this tutorial. In a new
    terminal, run `conda activate env_isaaclab` first.

## 1.3 Install Isaac Sim and PyTorch

```bash
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
python -m pip install --upgrade --force-reinstall torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

Start Isaac Sim once, accept the EULA, and wait until it has finished downloading extensions and
building shader caches (this can take more than ten minutes the first time). Then close it.

```bash
isaacsim
```

## 1.4 Install Isaac Lab

```bash
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2
./isaaclab.sh --install rsl_rl
```

## 1.5 Clone and install this project

Clone it next to (not inside) the `IsaacLab` directory, then install it in editable mode so your
code changes take effect without reinstalling:

```bash
cd ~
git clone https://github.com/LilSun-11/RL_SelfBalancingRobot_Training.git
cd ~/RL_SelfBalancingRobot_Training
python -m pip install -e source/SelfBalancing
```

All later commands are run from `~/RL_SelfBalancingRobot_Training` unless a `cd` says otherwise.

## 1.6 Install PlatformIO

[PlatformIO](https://platformio.org/) builds the ESP32 firmware and flashes it over USB (chapter 7).
Its official installer puts it in its own folder (`~/.platformio`), so it never touches the
packages of `env_isaaclab`:

```bash
cd ~
curl -fsSL -o get-platformio.py https://raw.githubusercontent.com/platformio/platformio-core-installer/master/get-platformio.py
python3 get-platformio.py
rm get-platformio.py
echo 'export PATH="$PATH:$HOME/.platformio/penv/bin"' >> ~/.bashrc
source ~/.bashrc
conda activate env_isaaclab
pio --version
```

## 1.7 Verify the installation

List the registered tasks. `Template-SelfBalancing-v0` and `Template-SelfBalancing-Play-v0` must
appear:

```bash
cd ~/RL_SelfBalancingRobot_Training
python scripts/list_envs.py
```

Run the task with zero actions. A window opens with many robots; with no torque on the wheels they
simply fall over and reset. That is expected: it shows the robot, the scene and the task load
correctly.

```bash
python scripts/zero_agent.py --task=Template-SelfBalancing-v0 --num_envs 16
```

!!! success "Checkpoint"
    Both tasks are listed, and the zero-action rollout opens without errors.
