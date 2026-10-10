# 7. Evaluate and Export

A rising reward curve is not proof that the robot behaves well. In this chapter you watch the trained
policy, judge it, and export it in a format the microcontroller can use.

!!! abstract "Learning objectives"
    - Play a checkpoint and read the debug arrows.
    - Decide whether a policy is ready for the real robot.
    - Export the policy to ONNX and check its input/output size.

## 7.1 Play a checkpoint

```bash
cd ~/SelfBalancing
python scripts/rsl_rl/play.py --task=Template-Selfbalancing-Play-v0 \
    --load_run <run_folder> --checkpoint model_<N>.pt
```

The Play task uses 360 robots, no sensor noise and the full command ranges from the start. Add
`--real-time` to run at real speed, or `--video --video_length 500` to record a clip to the run
folder instead of watching live. Without `--load_run`/`--checkpoint`, the latest checkpoint of the
latest run is used.

Each robot shows three arrows above it:

| Arrow | Meaning |
|---|---|
| Green | Target velocity (forward/backward) |
| Blue | Actual velocity |
| Orange | Target heading |

## 7.2 What to look for

Before taking a policy to the real robot, check:

- [ ] **No falls** after the first second (robots start tilted up to ±30°).
- [ ] **Standing still** when the green arrow is gone: the robot stays in place, with no slow drift
      or spinning.
- [ ] **Tracking**: the blue arrow follows the green one, and the robot turns to face the orange
      arrow without overshooting much.
- [ ] **Smooth wheels**: no buzzing back and forth. Jittery torque that the simulator tolerates
      often heats or stalls a real motor.
- [ ] **Recovery** from pushes, if `push_robot` is enabled.

Compare a few checkpoints of the same run (e.g. every 500 iterations): the latest is not always the
best.

## 7.3 Export the policy

`play.py` exports the loaded policy automatically, every time it runs:

```text
logs/rsl_rl/selfbalancing/<run_folder>/exported/
├── policy.pt      ← TorchScript (for Python / C++ with LibTorch)
└── policy.onnx    ← ONNX (used in chapter 8)
```

The exported network contains only the actor (no critic, no exploration noise): 10 inputs in the
order of chapter 5.8, 2 outputs.

Check the sizes before going further:

```bash
python -c "
import onnx
m = onnx.load('logs/rsl_rl/selfbalancing/<run_folder>/exported/policy.onnx')
dims = lambda t: [d.dim_value for d in t.type.tensor_type.shape.dim]
print('input', dims(m.graph.input[0]), 'output', dims(m.graph.output[0]))
"
```

It must print `input [1, 10] output [1, 2]`.

!!! success "Checkpoint"
    You have a `policy.onnx` with 10 inputs and 2 outputs from a checkpoint that passes the checks
    in 6.2.
