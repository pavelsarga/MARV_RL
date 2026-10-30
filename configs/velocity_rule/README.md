# V-Gov: rule-based forward-speed governor

V-Gov sets MARV's forward speed without a network. It starts from a constant cruise speed and
*governs* it down when terrain or robot state calls for care. It is the transparent counterpart to the
learned speed net (V-Net), and it also works without a map.

Steering is not its job. auto_ride's bearing law steers, and the flipper policy (SAS, A0t, …) runs
the flippers. V-Gov only sets `v`.

## How the speed is formed

Each tick, every criterion turns a metric into a ratio `r ∈ [0, 1]`. Its factor is `1 − weight·r`,
so the weight is the largest share of speed it can take away. A weight of 0.4 means "×0.6 when fully
triggered".

```
v* = v_default · Π (1 − weight_i · r_i)        compounded, then not below v_min
     predictive_combine: strongest → the map criteria contribute only their smallest factor, not their product
stall guard: measured v < stall.v while commanded ≥ v_min for stall.after_s → v* ≥ stall.boost for boost_s
slew:        v moves towards v* by at most accel·dt up, decel·dt down
```

How each criterion's ratio is set (`mode`):

- `linear`: `r = clip((m − lo) / (hi − lo))`, proportional. With `hold_s`, a peak is held that long
  before `r` follows the metric down again.
- `bang`: `r = 1` once `m > trip`. It drops back once `m < release` has lasted `hold_s`
  (hysteresis plus hold).
- `tau`: optional EMA smoothing of the metric, in seconds.

## Criteria

| name | part | metric |
|---|---|---|
| `step_up` / `step_down` | predictive | height change over `baseline` along the median track-footprint profile, from `lookahead` m ahead to `behind` m behind the centre. The map is world-fixed, so the slowdown holds until the edge has passed under the robot. |
| `roughness` | predictive | per scale s: s×s box average, then second difference at lag s. Planes give 0; s = 1 sees cell-to-cell variation, s = 4 sees ~20 cm patches. The `quantile` is taken under the robot, then the max over the scale weights. |
| `map_mismatch` | predictive | median height of the map above the robot's own contact plane, e.g. a tall-grass canopy. From `distrust_map_at`, the step and roughness criteria are ignored. |
| `tilt` | reactive | `hypot(roll_k·roll, pitch_k·pitch)`, in degrees |
| `shake` | reactive | body roll/pitch rate, in rad/s |
| `flippers` | reactive | fastest measured flipper, in rad/s |
| `turning` | reactive | commanded yaw rate, in rad/s |

`predictive: false` drops the map criteria, which is the mapless mode.

## Configs

- `default.yaml`: map plus reactive criteria.
- `mapless.yaml`: `extends: default.yaml` with `predictive: false`. Its reactive weights are
  stronger, so the compounded effect is similar, and shake has a short peak hold.

In Isaac, deep stacked slowdowns made MARV crawl over obstacles it climbs at speed. That is why
the defaults use `v_min: 0.2`, `predictive_combine: strongest`, roughness weight 0.15, tilt from
15°, and pitch at half the weight of roll.

Isaac results, Oct 2026: A0t on mixed_v2_straight, 16 envs × 5 repeats, `logs/policy_eval/rule_a0t/*_w3000`.
- Episodes have a 60 s timeout and the eval window is 3000 steps × 0.05 s = 150 s.
- Only each robot's first episode per repeat is scored. Later episodes still running when the window
  closes are never recorded, and scoring every episode biases the result toward fast controllers.
- Each row counts the 21 obstacle types that every run drew.
- Every row except native drives the Gazebo way: A0t's v / w are replaced, and the bearing law steers.

| A0t + … | success | failure | timeout |
|---|---|---|---|
| its own v / w | 0.886 | 0.068 | 0.045 |
| constant 0.41 m/s | 0.607 | 0.082 | 0.311 |
| V-Gov default | 0.593 | 0.130 | 0.278 |
| V-Gov mapless | 0.597 | 0.042 | 0.361 |
| V-Net (v2_s1, raw Isaac maps) | 0.325 | 0.013 | 0.662 |

Reading:
- The external speed / steering loop is the main cost. A0t stalls without its own v / w on stairs,
  tilted pallets, waves, twisted / bumpy hills and logs.
- Within that loop V-Gov matches a constant speed on success. Mapless fails least.
- Steering helps: without the bearing law, constant speed fell from 0.627 to 0.576 at a 30 s timeout,
  and failures doubled.
- An eval window shorter than the timeout drops stuck episodes. An earlier 50 s-window table showed
  V-Gov at ~0.9 that way, which was invalid.

## Running it

- **Gazebo / robot:**
  - `eval_run.py` with `speed_source: rule:<config name>` starts `velocity_rule_node`
    (robot_rodeo_gym), which publishes `/auto_ride/speed_cmd`.
  - Per tick it also publishes `/velocity_rule/state`, a JSON with v, target, each rule's factor and
    its metric. This topic is recorded in the bags.
  - In GUI runs, throttle_ui shows every rule's multiplier in a strip at the bottom.
  - Demo specs: `misc/velocity_il/specs/rule_*.json`.
- **Isaac:** `env_cfg_overrides.velocity_controller: {type: rule, config: configs/velocity_rule/default.yaml, history: 1}`.
  - Add `policy_actions: full` to run a policy trained to drive itself (e.g. A0t) the Gazebo way. It
    keeps its [v, w, flippers] action, and its v and w are replaced by V-Gov and the bearing law.
  - Set `track_vel_max: 1.0`, so the speed is not clamped at the 0.7 default.

Both simulators run the same file, `src/FTR-Benchmark/ftr_envs/tasks/crossing/velocity_rules.py`.
It is pure torch: Gazebo loads it by path and Isaac imports it. A parity test in
`test_velocity_rules.py` keeps the two input paths identical.

## Tuning

- The node reads the YAML at start-up, so restart the run after editing.
- `misc/velocity_il/rule_calibrate.py` prints the map criteria's metrics per area over the recorded
  rides. Use it to set `lo` / `trip` above the map's noise floor: roughness p95 ≈ 0.005 on flat
  Gazebo ground, and the map sits ~2 cm above the track plane on flat ground.
- Reactive thresholds are physical units and need no data.
