"""Run the firmware PID controller (Balancing_Robot_V2_PID/src/main.cpp) in Isaac Sim instead of a policy.

Reproduces the firmware's control law and motor shaper at the env's control rate (step_dt = 0.01s =
CONTROL_HZ 100), feeds the resulting signed duty to both wheels as the action (duty 1.0 <-> the
sim's MOTOR_TORQUE_MAX), and reports how many envs stay upright for a whole episode.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Balance the robot in sim with the firmware PID.")
parser.add_argument("--task", type=str, default="Template-SelfBalancing-Play-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--duration_s", type=float, default=20.5, help="Sim time to run (Play episode = 20s).")
parser.add_argument("--kp", type=float, default=0.4)
parser.add_argument("--ki", type=float, default=1.0)
parser.add_argument("--kd", type=float, default=0.01)
parser.add_argument("--i_max", type=float, default=0.6, help="Anti-windup limit on |Ki*integral| (duty).")
parser.add_argument("--min_duty", type=float, default=0.0, help="Firmware MIN_DUTY (only 0 is supported).")
parser.add_argument("--sign", type=float, default=1.0, help="Firmware MOTOR_SIGN equivalent (+1/-1).")
parser.add_argument("--init_tilt_deg", type=float, default=None, help="Override reset pitch range to +-this (deg), 0 init rate.")
parser.add_argument("--no_push", action="store_true", help="Disable the push_robot event.")
parser.add_argument("--no_delay", action="store_true", help="Set the wheel actuator delay to 0 steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import SelfBalancing.tasks  # noqa: F401

# firmware constants (main.cpp USER CONFIG)
MAX_DUTY, U_DEADBAND, LINEAR_START_U = 0.9, 0.01, 0.10
NEAR_BAL_MAX_DUTY, NEAR_BAL_PITCH_DEG, NEAR_BAL_RATE_DPS = 0.42, 1.5, 18.0


def shaper(u: torch.Tensor, near: torch.Tensor, last_dir: torch.Tensor) -> torch.Tensor:
    """Firmware shaperUpdate() for MIN_DUTY=0 (pulse-density branch inactive). Mutates last_dir.

    With REVERSE_COAST_S == dt, a direction reversal outputs 0 for exactly one tick.
    """
    abs_u = u.abs()
    dead = abs_u < U_DEADBAND
    req_dir = torch.where(u > 0, 1, -1).to(last_dir.dtype)
    reversing = (last_dir != 0) & (req_dir != last_dir) & ~dead
    lin_start = max(LINEAR_START_U, U_DEADBAND + 0.001)
    nrm = ((abs_u - lin_start) / max(0.001, 1.0 - lin_start)).clamp(0.0, 1.0)
    duty = nrm * MAX_DUTY
    duty = torch.where(near, torch.minimum(duty, torch.full_like(duty, NEAR_BAL_MAX_DUTY)), duty)
    duty = duty.clamp(0.0, MAX_DUTY)
    duty = torch.where(dead | reversing, torch.zeros_like(duty), duty)
    last_dir[:] = torch.where(dead, torch.zeros_like(last_dir), req_dir)
    return duty * req_dir.to(duty.dtype)


def main():
    if args_cli.min_duty != 0.0:
        raise ValueError("Only MIN_DUTY=0 (the firmware's current setting) is implemented.")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.init_tilt_deg is not None:
        t = math.radians(args_cli.init_tilt_deg)
        env_cfg.events.reset_base.params["pose_range"] = {"pitch": (-t, t)}
        env_cfg.events.reset_base.params["velocity_range"] = {"pitch": (0.0, 0.0)}
    if args_cli.no_push:
        env_cfg.events.push_robot = None
    if args_cli.no_delay:
        env_cfg.scene.robot.actuators["wheels"].min_delay = 0
        env_cfg.scene.robot.actuators["wheels"].max_delay = 0

    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dt = uenv.step_dt
    n = uenv.num_envs
    dev = uenv.device
    print(f"[PID] dt={dt}s kp={args_cli.kp} ki={args_cli.ki} kd={args_cli.kd} sign={args_cli.sign} "
          f"init_tilt={args_cli.init_tilt_deg} no_push={args_cli.no_push} no_delay={args_cli.no_delay}")

    env.reset()
    integral = torch.zeros(n, device=dev)
    last_dir = torch.zeros(n, dtype=torch.int32, device=dev)
    finished = torch.zeros(n, dtype=torch.bool, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    fall_time = torch.full((n,), float("nan"), device=dev)
    abs_pitch_sum = torch.zeros(n, device=dev)
    alive_steps = torch.zeros(n, device=dev)
    sat_steps = 0.0
    total_steps = 0.0
    # sign check: d(pitch)/dt (finite difference) vs the sim's pitch_rate obs (root_ang_vel_b[:, 1])
    corr_num, corr_den_a, corr_den_b = 0.0, 0.0, 0.0
    prev_pitch = None

    steps = int(round(args_cli.duration_s / dt))
    with torch.inference_mode():
        for k in range(steps):
            pg = robot.data.projected_gravity_b
            pitch = torch.atan2(-pg[:, 0], -pg[:, 2])  # == mdp.imu_pitch_angle
            ang_y = robot.data.root_ang_vel_b[:, 1]  # == mdp.imu_pitch_rate
            pitch_deg = torch.rad2deg(pitch)
            # firmware's rateDps is d(pitch)/dt (it integrates it in the complementary filter)
            if prev_pitch is not None:
                d_pitch = (pitch - prev_pitch) / dt
                valid = ~just_reset
                corr_num += float((d_pitch * ang_y)[valid].sum())
                corr_den_a += float((d_pitch**2)[valid].sum())
                corr_den_b += float((ang_y**2)[valid].sum())
            prev_pitch = pitch.clone()
            rate_dps = torch.rad2deg(-ang_y)

            error = -pitch_deg
            if args_cli.ki > 1e-6:
                integral = (integral + error * dt).clamp(-args_cli.i_max / args_cli.ki, args_cli.i_max / args_cli.ki)
            else:
                integral.zero_()
            u = (args_cli.kp * error + args_cli.ki * integral - args_cli.kd * rate_dps).clamp(-1.0, 1.0)
            near = (pitch_deg.abs() <= NEAR_BAL_PITCH_DEG) & (rate_dps.abs() <= NEAR_BAL_RATE_DPS)
            duty = shaper(u, near, last_dir)

            live = ~finished
            abs_pitch_sum += torch.where(live, pitch_deg.abs(), torch.zeros_like(pitch_deg))
            alive_steps += live.float()
            sat_steps += float(((duty.abs() >= MAX_DUTY - 1e-6) & live).sum())
            total_steps += float(live.sum())

            actions = (args_cli.sign * duty).unsqueeze(1).repeat(1, 2)
            _, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated

            newly_fell = done & terminated & ~finished
            fell |= newly_fell
            fall_time = torch.where(newly_fell, torch.full_like(fall_time, (k + 1) * dt), fall_time)
            finished |= done
            integral = torch.where(done, torch.zeros_like(integral), integral)
            last_dir = torch.where(done, torch.zeros_like(last_dir), last_dir)
            just_reset = done

    survived = finished & ~fell
    unfinished = ~finished
    print("=" * 70)
    print(f"[PID] envs={n}  survived full episode={int(survived.sum())}  fell={int(fell.sum())}  "
          f"still running={int(unfinished.sum())}")
    if fell.any():
        ft = fall_time[fell]
        print(f"[PID] fall time (s): mean={ft.mean():.2f} min={ft.min():.2f} max={ft.max():.2f}")
    mean_abs_pitch = (abs_pitch_sum / alive_steps.clamp(min=1)).mean()
    print(f"[PID] mean |pitch| while alive = {mean_abs_pitch:.2f} deg,  duty saturated "
          f"{100.0 * sat_steps / max(total_steps, 1):.1f}% of alive steps")
    if corr_den_a > 0 and corr_den_b > 0:
        corr = corr_num / math.sqrt(corr_den_a * corr_den_b)
        print(f"[PID] corr(d(pitch_angle)/dt, pitch_rate obs) = {corr:+.3f}  "
              f"({'SAME sign' if corr > 0 else 'OPPOSITE sign'})")
    print("=" * 70)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
