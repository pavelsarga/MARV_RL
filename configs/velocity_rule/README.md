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

**Stall guard (`stall`).**
- A ring of poses gives the progress along the heading over the last `window_s`. Less than
  `progress_m` while commanding ≥ `v_min` is a stall.
- The response overrides the rules at `boost[level]` until `release_m` of progress, or until
  `max_boost_s` runs out. When a boost runs out, the next level starts at once. Past the top level
  the guard waits a fresh window; getting through resets the level.
- Thresholds come from the tile-sweep trajectories. Rocking in place covers ≤ 0.20 m per 3 s (p99),
  so `progress_m: 0.25` and `release_m: 0.4`.
- `max_roll_deg` (optional, off) keeps a rolled robot at the first level.
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

Current defaults (Nov 2026): 0.1–0.4 m/s (`v_min` / `v_default`), stall boosts `[0.2, 0.3, 0.4]` (the
top one = `v_default`).
- **Weights.** They are the most conservative set that keeps the speed on the floor for at most 25 % of a
  ride, picked for the slowest descents.
- **How they were picked.** Three recorded Gazebo rides were replayed offline through the node's own
  input path: the GT map gives the robot window and the odometry gives the attitude. The criteria's
  ratios do not depend on the weights, so each candidate set is exact and costs nothing.
- **Result on those rides.** Floor 23–24 %, mean 0.189 m/s, 0.150 m/s while pitched nose-down.
- **Descents.**
  - The robot used to speed up on descents. step_down released once the edge was 0.10 m behind the
    centre, while the rear was still going down. Nose-down pitch counted at half weight, so a 30°
    descent was below tilt's `lo`. And the stall guard boosted a robot creeping over the lip.
  - Now step_down's `behind` is 0.45 m (past the rear of the footprint), and `pitch_down_k: 1.0` counts
    descents in full; climbs stay at `pitch_k` 0.5.
  - The stall guard's `window_s` is 5 s (at 3 s, floor-speed creeping read as a stall and was boosted
    32 % of a ride; 8 % at 5 s). `max_pitch_down_deg: 10` keeps a nose-down robot at the first boost level.
- **`flippers`** is latched about 63 % of the time (the policy moves its flippers almost constantly), so
  its weight acts nearly everywhere.
- **`mapless.yaml`** cannot see steps. It raises flippers and shake to 0.7 to match default's mean speed.
- `predictive_combine: strongest`, tilt from 15° and pitch at half the weight of roll stay: in Isaac,
  deep stacked slowdowns made MARV crawl over obstacles it climbs at speed.

Isaac training with V-Gov: `configs/diffusion/marv_config_diffusion_p1_flipperonly_vgov_ss_x3t.yaml`.

Isaac results, Nov 2026, from `eval_diffusion.py --tile_sweep` (`logs/policy_eval/tile_sweep_rect/`), with the
**previous** defaults (0.2–0.8 m/s, lighter weights, boosts `[0.4, 0.6, 0.8]`).

Setup:
- The A0t flippers at their trained 10 Hz, on all 350 tiles of mixed_v2_straight (35 obstacles × 10
  columns, forward, centre spawn).
- Each controller drives each tile once, run to success, failure or a 120 s timeout.
- Success means the target line is reached anywhere across the spot (`success_shape: rectangle`,
  ±1.67 m). Failures are split by `fail_reason`.
- Every row except native drives the Gazebo way: A0t's v / w are replaced, and the bearing law steers.

| A0t + … | success | timeout | failure | rollover | out of range sideways | median success time |
|---|---|---|---|---|---|---|
| its own v / w | 0.977 | 0.006 | 0.017 | 0.006 | 0.011 | 13.8 s |
| constant 0.41 m/s | 0.806 | 0.183 | 0.011 | 0 | 0.011 | 20.4 s |
| V-Gov default | 0.943 | 0.040 | 0.017 | 0.003 | 0.014 | 19.4 s |
| V-Gov default, `distance_ramp` steering | 0.937 | 0.037 | 0.026 | 0 | 0.026 | 19.8 s |
| V-Gov mapless | 0.943 | 0.040 | 0.017 | 0.009 | 0.009 | 25.0 s |

Reading:
- In the same loop V-Gov beats a constant speed by 14 points, almost all of it stalls: 18 % → 4 %,
  from the progress stall guard (stairs 0–3/10 → 9–10/10). Failures stay as low as native's.
- The remaining 3 points to native are the stalls A0t avoids by driving its own speed.
- Mapless matches default but is slower.
- Weaker steering while crossing (`heading_controller.distance_ramp`) doubles the sideways
  out-of-range and gains nothing, so keep full steering.

Earlier tables in this file's history are invalid:
- `eval_diffusion` ran diffusion policies at 4× their trained rate.
- Fixed eval windows cut episodes off and gave each controller different tiles.
- The old 0.4 m success circle counted a robot that crossed but drifted half a metre sideways as a
  failure. That was ~90 % of all failures, for every controller.

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
