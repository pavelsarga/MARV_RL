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

Isaac results, Oct 2026, from `eval_diffusion.py --tile_sweep` (`logs/policy_eval/tile_sweep_a0t/`):
- The A0t flipper policy at its trained 10 Hz, on all 350 tiles of mixed_v2_straight (35 obstacles
  × 10 columns, forward, centre spawn).
- Every controller drives exactly those tiles, one episode each, run to success, failure or a 120 s
  timeout. Nothing is cut off and nothing is dropped.
- Every row except native drives the Gazebo way: A0t's v / w are replaced, and the bearing law steers.

| A0t + … | success | failure | timeout | median success time |
|---|---|---|---|---|
| its own v / w | 0.746 | 0.226 | 0.029 | 13.0 s |
| constant 0.41 m/s | 0.709 | 0.120 | 0.171 | 19.2 s |
| V-Gov default | 0.654 | 0.097 | 0.249 | 16.4 s |
| V-Gov mapless | 0.657 | 0.120 | 0.223 | 22.6 s |
| V-Net (v2_s1, raw Isaac maps) | 0.546 | 0.100 | 0.354 | 16.6 s |

Reading:
- The bearing law makes easy terrain safe. Flat ground, ramps, gaps and trenches go from 7–9/10 to
  10/10.
- Every loop stalls where A0t needs its own speed: stairs, log crossing, diagonal log / trunk /
  pyramid, tilted pallet, wave, and the twisted / bumpy hills.
- V-Gov has the fewest failures (default 9.7 %) but stalls more than a constant speed. On lowered /
  steep stairs it makes 0/10 against 2/10, so its step rules take away momentum the climb needs.
- It is better on steep hills (10/10) and fails less on the wave.

Earlier tables in this file's history are invalid. `eval_diffusion` ran diffusion policies at the
env's default decimation, 4× their trained rate. Its fixed eval windows also cut episodes off and
spread tiles unevenly across controllers.

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
