# 9. Troubleshooting

## Installation and simulation

| Symptom | Cause and fix |
|---|---|
| `python` points outside `env_isaaclab`, or `isaacsim` / `isaaclab.sh` not found | Run `conda activate env_isaaclab` and retry |
| `ModuleNotFoundError: isaaclab` or `isaaclab_rl` | Rerun `./isaaclab.sh --install rsl_rl` from the `IsaacLab` folder |
| `torch.cuda.is_available()` is `False` | Repair the NVIDIA driver, then reinstall the CUDA 12.8 PyTorch command from 1.3 |
| Task not listed by `scripts/list_envs.py` | Run `python -m pip install -e source/SelfBalancing` in `~/SelfBalancing`; the task id must start with `Template-` |
| First Isaac Sim launch looks frozen | Wait: the first launch downloads extensions and builds shader caches (10+ minutes) |
| GPU out of memory | Lower `--num_envs` |
| Errors about `actor` / `critic` / `policy` fields in the RL config | A different `rsl-rl-lib` version; use Isaac Lab v2.3.2 with RSL-RL 3.1.2 |
| `ModuleNotFoundError` / `AttributeError: module ... has no attribute` for a term | A file in `mdp/` is missing, or `mdp/__init__.py` does not import it (chapter 5.2) |
| Robot sinks into or floats above the ground at spawn | Spawn height in `robot.py` does not match the URDF (chapter 3.6) |
| `size mismatch` when resuming | The number of observations or actions changed since that checkpoint; start a new run |
| Training explodes (reward −10¹⁸, `std >= 0.0` error) | Set `clip_actions = 1.0` (chapter 6.6) |

## Firmware and real robot

| Symptom | Cause and fix |
|---|---|
| `Permission denied: /dev/ttyUSB0` | Do 8.3 (udev rules + `dialout` group), then log out and back in |
| No `/dev/ttyUSB*` appears | Charge-only cable → use a data cable. CH340 board → `sudo apt remove brltty` and replug |
| `pio: command not found` | `source ~/.bashrc`, or redo 1.5 |
| Upload stuck at `Connecting...` | Hold the **BOOT** button until the upload starts |
| Port busy | Close the serial monitor (`Ctrl + C`) |
| Garbled serial output | Open the monitor with `pio device monitor` from the project folder (921600 baud) |
| Compile error about `POLICY_OBS_DIM` | The policy's observation count differs from `assembleObs()`; regenerate the header from a 10-observation policy |
| `--verify` fails | The model is not a plain MLP; re-export it with `play.py` |
| `# LOI: khong thay LSM6DS3` and the LED blinks red | IMU not found: check SDA 21 / SCL 22, 3V3 and GND |
| Robot falls immediately | Check the observation signs (8.10 steps 1–3), then the pitch trim (`s`) |
| Robot spins in place | Yaw sign (`z-1`) or wheels swapped (check with `j`, fix with `w`) |
| Robot slowly rolls away while balancing | Pitch offset: recalibrate (`c`) with the robot exactly upright, or adjust `s` |
| Sensor values jump when the motors run | Missing common ground, loose wires or a low battery |
| ESP32 resets when the motors start (`BROWNOUT`) | Supply voltage drops: check the buck converter, add capacitance, check the ground |
