# Firmware

ESP32 firmware for the real robot. There are two PlatformIO projects:

| Project | What it runs | Serial baud |
|---|---|---|
| [`policy_controller/`](policy_controller/) | The RL policy trained in this repo (MLP from `include/policy_weights.h`), one torque command per wheel | 921600 |
| [`pid_controller/`](pid_controller/) | A classic PID on the tilt angle, same command for both wheels (baseline) | 115200 |

Both run a 100 Hz control loop and share the same sensor, angle-filter and motor-driver code; only
the controller differs. [`scripts/pid_balance.py`](../scripts/pid_balance.py) runs the same PID law
and motor shaper in Isaac Sim, which is useful for checking whether the simulator matches the robot.

## Hardware

- ESP32 DevKit (`board = esp32dev`)
- LSM6DS3 IMU over I2C (address 0x6B or 0x6A, detected automatically)
- Dual H-bridge motor driver (enable + 2 direction pins per motor)
- 2 × JGB37-520 geared DC motors with quadrature encoders (11 PPR × 30:1 gearbox × 4 = 1320 ticks
  per wheel revolution), 65 mm wheels
- Addressable RGB status LED

| Function | ESP32 pin |
|---|---|
| IMU SDA / SCL | GPIO 21 / 22 |
| Left motor (A): enable (PWM) / IN1 / IN2 | GPIO 15 / 2 / 4 |
| Right motor (B): enable (PWM) / IN3 / IN4 | GPIO 5 / 19 / 18 |
| Left encoder A / B (policy firmware only) | GPIO 14 / 26 |
| Right encoder A / B (policy firmware only) | GPIO 33 / 32 |
| Status LED | GPIO 27 |

LED: green = running, dim red = motors disabled, bright red = tilted past the safety limit (motors
cut): 60° for `policy_controller`, 45° for `pid_controller`.

## Build and flash

Install [PlatformIO](https://platformio.org/install/cli), then from the project folder:

```bash
cd firmware/policy_controller          # or firmware/pid_controller
pio run -t upload                      # build and flash (dependencies download on first build)
pio device monitor                     # serial monitor, baud taken from platformio.ini
```

On boot the firmware calibrates the IMU for ~2 s: **hold the robot upright and still** until it
prints `# CALIB offset=...`.

## Deploying a new policy

1. Train and evaluate in Isaac Sim. `scripts/rsl_rl/play.py` writes
   `logs/rsl_rl/selfbalancing/<run>/exported/policy.onnx` (see the main [README](../README.md)).
2. Convert it to a C header with the converter in [`tools/`](../tools/):

   ```bash
   pip install -r tools/requirements.txt
   python tools/export_policy_header.py \
       --model logs/rsl_rl/selfbalancing/<run>/exported/policy.onnx \
       --out firmware/policy_controller/include/policy_weights.h --verify
   ```

   `--verify` compares the header's C-style forward pass against ONNX Runtime and fails loudly on
   any mismatch. The converter only accepts a straight-chain MLP (Linear + ELU/ReLU/Tanh/Sigmoid).
3. Rebuild and flash `policy_controller`.

The firmware refuses to compile unless the header has `POLICY_OBS_DIM == 10` and
`POLICY_ACT_DIM == 2`, matching the observation layout built in `assembleObs()`:

```text
[pitch_angle, pitch_rate, yaw_angle, yaw_rate, wheel1_vel, wheel2_vel,
 last_action1, last_action2, velocity_command, yaw_command]
```

This must stay in sync with `ObservationsCfg.PolicyCfg` in `selfbalancing_env_cfg.py` (order, units
and sign convention — see [Policy input/output](../README.md#policy-inputoutput)). In particular:

- the sim's `pitch_rate` is the negative of d(pitch)/dt, so the firmware feeds `-rateDps`;
- `yaw_rate` is the gyro Z rate (rad/s, + = turning left), bias-corrected at calibration and
  low-pass filtered at `YAW_LPF_HZ` (20 Hz, `f` command);
- `yaw_angle` is the unfiltered gyro Z rate integrated since the last calibration or `h` command,
  wrapped to [-π, π] — the real-robot equivalent of the sim's spawn heading = yaw 0. There is no
  magnetometer, so it slowly drifts;
- `yaw_command` is a target heading in that same frame (`y<deg>`). In the sim, a robot told to stand
  still (`v = 0`) holds the heading it has at that moment, so to stand still on the real robot set
  `y` to the current `yawAng`.

## Motor output shaping

The controller output `u ∈ [-1, 1]` is turned into a PWM duty by `shaperUpdate()`:

- `|u| < U_DEADBAND` → motor off.
- Linear map from `LINEAR_START_U` up to `MAX_DUTY` (0.9). With `MIN_DUTY > 0`, small commands
  are sent as short pulses at `MIN_DUTY` (pulse-density modulation) to get past motor stiction.
- Near balance (`|pitch| ≤ 1.5°` and `|rate| ≤ 18°/s`) the duty is capped at `NEAR_BAL_MAX_DUTY`.
- A change of direction coasts the motor for one control tick.

## Serial commands

Send one command per line, e.g. `p0.4` or `m-1`. `?` prints the current parameters.

| Command | Both firmwares |
|---|---|
| `a<v>` | Output scale `U_SCALE` (0–1) |
| `n<v>` / `l<v>` / `b<v>` | `MIN_DUTY` / `MAX_DUTY` / `U_DEADBAND` |
| `m<±1>` | Motor direction sign |
| `g<±1>` / `i<±1>` | Gyro sign / pitch sign (flip if the IMU is mounted the other way) |
| `c` | Re-calibrate the IMU (hold the robot upright) |
| `t` | Motor test: both wheels forward then backward at 50% for 1.5 s each |
| `x` / `o` | Motors off / on |

| Command | `policy_controller` only |
|---|---|
| `s<deg>` | Pitch trim (deg) |
| `e<±1>` / `r<±1>` | Left / right encoder count direction |
| `w<0/1>` | Swap the two policy outputs between the wheels |
| `v<m/s>` | Velocity command, + = forward, clamped to ±`VEL_CMD_MAX` (0.1 m/s) |
| `y<deg>` | Heading (yaw) command relative to yaw 0, + = left (e.g. `y30`) |
| `h` | Make the current heading yaw 0 and clear the heading command |
| `z<±1>` | Gyro Z sign (turning the robot left by hand must give a positive `yaw`) |
| `f<Hz>` | Yaw-rate low-pass cutoff (0 = off) |
| `j` | Wheel mapping test (wheels off the ground): drives `act[0]` then `act[1]` alone and checks with the encoders that only the left, then only the right wheel turns forward |

| Command | `pid_controller` only |
|---|---|
| `s<deg>` | Pitch setpoint (deg) |
| `p<v>` / `k<v>` / `d<v>` | Kp / Ki / Kd (changing Ki resets the integral) |
| `<kp>,<ki>,<kd>` | Set all three gains at once, e.g. `0.4,1.0,0.01` |

Defaults of the PID firmware: `Kp = 0.4` duty/deg, `Ki = 1.0` duty/(deg·s) with anti-windup
`|Ki·∫e| ≤ 0.6`, `Kd = 0.01` duty/(deg/s) applied to the measured rate.

## First run checklist

1. Run `t` with the wheels off the ground: both wheels must spin the same way, forward first. If
   not, swap the motor wires on one side or use `m-1`.
2. Tilt the robot by hand and watch the telemetry: pitch and rate must move in the same direction
   (`g`, `i` to fix), and each encoder velocity must be positive when its wheel rolls forward (`e`,
   `r` to fix). Turn the robot left (counter-clockwise seen from above): `yaw` (rate) must be
   positive and `yawAng` must increase (`z-1` to fix).
3. Run `j` with the wheels off the ground: it must report `OK` for both actions (`w1`/`w0` swaps the
   wheels, `e`/`r` the encoder directions).
4. Only then put it on the floor, holding it lightly for the first seconds.
