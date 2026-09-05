# 07 — ROS2 deployment (design now, build later)

Not implemented in the first pass. This records what the design has to accommodate so we
do not paint ourselves into a corner.

## Current path

`ros2/flipper_policy_node.py` runs a control loop at `control_rate` (default 10 Hz,
matching training) and calls into `training/ftr_policy_inference_module.py`, which builds
the observation and runs the actor.

Two conveniences already line up with our choices:

- The node **already publishes flipper commands as positions**
  (`/flippers_cmd_pos/{front_left,front_right,rear_left,rear_right}`, `std_msgs/Float64`),
  so the `flipper_control_mode: position` decision matches the deployment interface
  rather than fighting it.
- The node's control rate is 10 Hz, the same as training, so `T_a` means the same number
  of wall-clock milliseconds in both places.

## What receding horizon adds

Two pieces of state, both of which belong **inside the inference module**, not the node,
so that `eval_ftr.py` and the ROS node share one implementation and cannot drift:

1. A `T_o`-deep observation deque, padded with the first frame on start (`padding="same"`,
   matching `CatFrames`).
2. A chunk cursor: run inference on tick `i` when `i % T_a == 0`, cache the `[T_p, A]`
   chunk, and replay `chunk[i % T_a]` on the intermediate ticks.

## Things to get right when we build it

- **Latency placement.** The paper's latency ablation measures the gap between the last
  observation and the first executable action. If inference takes longer than one tick,
  the chunk should be planned to start at the tick it will actually be ready for, not the
  tick it was requested on. Budget: `K_infer * per-step` must fit inside `T_a * 100 ms`
  (see `03_network.md`).
- **Startup.** The first `T_o - 1` ticks have no history. Pad, do not stall.
- **Abort.** A chunk is open-loop by construction. The node needs a way to drop the
  remaining chunk and re-plan immediately on an operator stop or a safety trip. Decide
  whether that is a hard zero or a replan.
- **Heightmap decay.** The node applies `heightmap_decay` (0.95) temporal smoothing that
  training does not have. With a `T_o` window stacking already-smoothed frames, the
  effective history is longer than `T_o` ticks. Worth checking whether `T_o=2` on the real
  robot is doing anything at all.
