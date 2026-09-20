# Terrain v2 — why `custom_mixed` was replaced, and what by

*Written 2026-09-19. Generator: `src/FTR-Bench-terrain-gen`; courses:
`terrain_config_marv_calib.yaml`, `terrain_config_mixed_v2_{straight,full,holdout}.yaml`;
training configs: `configs/terrain_v2/`.*

## 1. What was wrong with `custom_mixed`

| symptom | evidence |
|---|---|
| Saturated | best run (`logs/diff_p1_Ta1_11494827/attempt_6`) 0.946 overall; **columns 0–3 ≈ 100 % on every one of the 17 rows** — 40 % of the course is free reward. |
| Full-throttle policy | `action/v_mean = 0.955 ± 0.09`; nothing in the course punishes speed enough to make slowing down a decision. |
| Handed policy | `diagonal_trunk` is graded −45° → +45° across the row; the same geometry mirrored scores **0.97 at −45° and 0.31 at +45°**. |
| Difficulty ≠ column | `cobblestones` scores 0.39 at col 7 and 0.71 at col 8: the per-tile seed decides the layout, `t` only scales the max height. |
| All boxes | every obstacle is `add_ground_slab`/`add_cube`; there was not a single sloped surface, so the flippers were never required to *modulate*, only to be set once per obstacle. |
| One approach per tile | one spawn per tile, always −X, always on the centreline. |

## 2. MARV's physical envelope

From `robot_rodeo_gym_ws/.../marv/config/description/common.yaml` and `marv.xacro`
(the same numbers the sim USD is built from):

| quantity | value |
|---|---|
| body | 0.605 × 0.360 × 0.236 m |
| flipper pivots | x = ±0.256 (pivot spacing 0.512), y = ±0.248 (gauge 0.496, overall width 0.571) |
| flipper | 0.498 long; wheels r 0.1165 (pivot) → 0.078 (tip) over 0.3035; tip reach 0.3815 |
| footprint, flat | 1.119 m axle-to-axle, 1.275 m tip-to-tip |
| belly clearance | ≈ 0.07 m flat; ≈ 0.33 m with all flippers 80° down |
| flipper limits (cfg) | 60° up / 80° down |
| mass | ≈ 50 kg |
| speed (sim) | 0.7 m/s, w ≤ 1 rad/s, yaw authority ≈ 4 % under load |
| heightmap obs | 2.25 × 1.05 m at 5 cm — 0.49 m past the flipper tip, ±0.525 m laterally |

Reference limits from `cur_mixed` heights (`map/cur_mixed.map`) × the cur_mixed-trained
`experiments/thesis/best_long` per-spot success. That eval ran with the **FTR ("pumbaa")
robot** (its `eval_ftr_11014024.out` spawns `/World/envs/env_0/pumbaa_wheel/…`; no
`robot_type` in the config), which is ~1.08× MARV with main tracks — so these are an **upper
bound** for MARV: whatever FTR could not cross, MARV cannot either, and the calibration course
(§7) can only move the MARV limits *down* from here, never up.
```
single step-up      .30 .30 .35 .35 .35 .40 .40 .40 .40 .40 | 1 1 1 .97 .91 .84 .60 .33 .10 .00
pit (drop+climb)    .30 .30 .35 .35 .40 .40 .45 .45 .50     | 1 1 1 1 .26 0 0 0 0 0
trench climb-out    .20 .20 .25 .25 .30 .30 .35 .35 .40     | 1 1 1 .99 .95 .98 .96 .37 0 0
mound               .45 .50 .55 .60 .65 .70 .70 .75 .85 .85 | .9 .9 .8 .82 .6 .81 .04 .49 .05 .28
stairs (4 steps)    .60 .65 .65 .70 .70 .75 .80 .80 .85 .85 | ≥.97 until .85 → .21
0.40 m block fields random layouts                          | 0.00 – 0.94 (layout lottery)
```

→ upper bounds for MARV: step-up **≤ 0.35** (0.40 is a wall even for FTR), pit climb-out
**≤ 0.35**, mound **≤ 0.60**, stairs **≤ 0.20/step**, 0.40 m random block fields unreliable.
MARV's own flipper reach is comparable (tip axle at 0.379 m with the front pair at 60°), so
the v2 top columns were set at these bounds; the calibration course extends every class
past them so §7 can show where MARV actually falls short of FTR.

## 3. Course design

### Grading
Every kept row is re-graded so column 0 ≈ the old column 4 and column 9 sits just inside the
limit above (`raised_platform` 0.22 → 0.36, `pit` 0.20 → 0.32, stairs 0.12 → 0.20 per step,
fields ≤ 0.30 with a neighbour rule, pyramid ≤ 0.45). Rows can also set
`grade: sqrt|quadratic` to spend more columns on the hard end.

### New obstacle classes (all in `ftr_terrain_gen/obstacles/`)

| row | what it forces | source |
|---|---|---|
| `pitch_ramp` A / V | posture on a slope; A crest high-centres the 1.12 m track, V trough grounds the belly (≈ 7° breakover with flat flippers) | NIST inclined plane; Číhala A/U ramps |
| `diagonal_ramp` | pitch + roll at once; asymmetric flipper timing | FTR-Bench rails 10–45° |
| `diagonal_stairs` | one track meets each riser first | RoboCup stairs with debris |
| `cross_slope` | left/right flippers do different things (helicoid twist-in, no entry step) | NIST 15° roll ramp |
| `crossing_ramps` | anti-phase wedges left/right: pitch+roll sign flips every 0.6 m | NIST crossing pitch/roll ramps |
| `wave` | continuous modulation, optional yaw | FTR-Bench wave terrain |
| `round_log` / `diagonal_log` | no edge to hook: press-and-roll; diagonal → one side first | RoboCup pipes/rails |
| `lowered_platform` (pit) | drop, re-hook the far wall with the rear pair still in the pit | cur_mixed lowered_platform |
| `deep_gap` (widening_trench, 0.6 deep, 0.15–0.45 wide) | bridge with flat flippers, speed control; falling in ends the episode | NIST gap |
| `drive_trench` (widening_trench, 0.25 deep, 0.6–1.5 wide) | shallow drive-through — kept strictly wider than `deep_gap` | — |
| `stepfield` | NIST pallet: 0.15 m posts, 4 levels, neighbour Δ ≤ 2 units, flat/hill/cross/diagonal-hill layouts — traversable by construction | Jacoff 2008 |
| `tilted_pallet` | ramp that is not square to the robot, drop that is not square either | Číhala tilted, partly-buried pallets |
| `sequence` | two short features ~1.9 m apart in one tile — approach speed to the second is a decision | — |
| `offset_gate`, `chicane`, `flat_turn` (full course only) | turning, with the gate opening kept inside the ±0.525 m heightmap window | RoboCup traverse-and-center |
| `steep_hill` | **stability, not tilt avoidance**: 1.4–1.8 m ramps at 20–38° (0.51–1.41 m tall, both length and angle grade) put the whole 1.12 m track on the slope; the body must stay pitched for seconds with the rear flippers behind the CoM, and the 0.6 m crest (shorter than the robot) is a transition, not a rest. `n_hills` chains hills that grow along the path (`hill_growth`) | NIST 30–45° inclined planes; RoboCup ramps |
| `twisted_hill` | the same hill with a roll that changes sign along the path (1.5 sine periods, ±6–14°): pitch and a reversing roll at once | — |
| `bumpy_hill` | the same hill with 16 seeded-random cosine bumps/hollows (0.04–0.12 m, r 0.25–0.45) inside the driving band — disturbances that cannot be timed | rubble slopes |

`log_crossing` (the 0.25 m-wide beam the robot straddles belly-first) is kept as the
"obstacle smaller than the robot" class — Číhala et al. found it the hardest for autonomous
policies — so no separate hurdle wall was added.

### Course-level augmentation (generated into `birth.json`, no consumer change)
- `bidirectional: true` — every tile is also driven in reverse (RoboCup scores every lane
  end-to-end both ways); a one-way stair/ramp/pit is a different obstacle backwards.
- `spawn_lateral_jitter` — spawns at y = 0, ±0.15 (holdout: ±0.20).
- `mirror_alternate: true` on every asymmetric row — odd columns use the opposite
  handedness at the same difficulty.
- `goal_lateral_offset` (full course only) — target shifted behind the gate opening.

`env_type_registry` locates a tile by rounding the *target* to the nearest tile centre, so all
of the above maps back to the right (row, col) unchanged; a row's optional `name:` labels the
env type (e.g. `steep_stairs`, `deep_gap`, `log_then_step`).

### The three courses
| course | rows | spawn entries | purpose |
|---|---|---|---|
| `mixed_v2_straight` | 31 | 1860 | training, straight-drive only |
| `mixed_v2_full` | 34 (= straight + 3 turning rows, by `include_rows`) | 2040 | training, needs pivoting |
| `mixed_v2_holdout` | 29 | 1740 | generalization eval only: every training class, varied (in-between heights, other seeds/treads/post sizes, opposite handedness order, two new sequences) |
| `marv_calib` | 9 | 180 | not for training: each class graded past its known limit |

Evaluate any saved policy on another course with `--map <course>`
(`bash scripts/eval_auto.sh <run_dir> --map mixed_v2_holdout ...`).

## 4. Spawn-pointer facts (the "unequal crossings" question)

`FtrEnv._prepare_reset_info` (`ftr_env.py:813-826`) hands spawns out with `itertools.cycle`
over `birth.json` in file order; every spawned robot takes the next entry, so *started*
episodes per tile equalise to ±1. What is unequal:
- *completed* episodes inside a fixed eval window skew to easy tiles (hard tiles are still
  running when the window closes) — cur_mixed evals show 1473–1977 per row from 10 spawns each;
- *frames* per tile: a timed-out tile contributes ~4–6× the frames of one solved in 5 s.
Re-grading (no trivial columns) and pairing short features address most of it; a
success-rate-weighted sampler in `FtrEnv` is the literature's next step (LP-ACRL 2026) and is
not built here.

## 5. Turn (friction) test — result

`scripts/turn_test.sh <ppo_run_dir>` holds each flipper posture with
`env_cfg_overrides.flipper_control_mode=position` and commands w = +1 (= `track_ang_vel_max`,
tracks at ±0.18 m/s) open-loop on `ground`, 16 robots × 600 steps (7.5 s), via
`eval_ftr.py --scripted_action`. `state/yaw_rate` is the signed mean over all robots and
steps, `heading_err_deg_min` the largest heading change reached (2026-09-19, MARV,
`experiments/baselines/marv_rl_2` physics: wheel μ 4.0, terrain μ 1.2/0.9):

| posture | v = 0: mean yaw (rad/s) / max heading change | v = 0.5: mean yaw / max heading change |
|---|---|---|
| flat | 0.118 / −65° | 0.052 / −55° |
| front pair up 60° | 0.073 / −60° | 0.061 / −59° |
| rear pair up 60° | 0.124 / −94° | 0.102 / −85° |
| **both pairs up** | **0.185 / −123°** | 0.073 / −76° |
| all four down 80° ("tiptoe") | 0.132 / −93° | 0.101 / −76° |

Kinematic reference with the real 0.496 m gauge: 0.73 rad/s — so MARV realises 15–25 % of it
when pivoting and 7–14 % while driving. That is slow (a 15° turn takes 1.5–2.5 s standing, or
~4 m of travel at v = 0.5), but it is not zero: the CLAUDE.md "3.6 %" figure was measured
under load on obstacles with a saturated heading controller. The gate rows only need 8–15° of
heading change on a flat approach, so `mixed_v2_full` is physically solvable; the cheapest
pivot is with both flipper pairs raised (pivoting on the big wheels), which is the posture
change we want the policy to discover. Raw logs: `logs/turn_test/`.

## 6. Calibration result (MARV, `marv_calib`, 96 robots × 2 repeats, ~5 episodes per spot)

`experiments/baselines/marv_rl_2` (PPO, 0.916 on custom_mixed) and `logs/diff_p1_Ta1_11494827/attempt_6`
(diffusion, 0.946), both trained on custom_mixed, both driven forward and in reverse:

```
                     col:  0    1    2    3    4    5    6    7    8    9
step_up  (0.20→0.50)  Ta1  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0    heights .20 .23 .27 .30 .33 .37 .40 .43 .47 .50
                      rl2  .75  .60  .60  .50  0.0  0.0  0.0  0.0  0.0  0.0
pit      (0.20→0.50)  Ta1  1.0  1.0  .75  .50  0.0  0.0  0.0  0.0  0.0  0.0
stairs   rise .10→.24 Ta1  .83  .75  .50  .25  0.0  0.0  0.0  0.0  0.0  0.0    tread 0.28
deep_gap (0.15→0.60)  Ta1  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0    widths .15 … .45 | .50 .55 .60
ramp_a   (5→40°)      Ta1  1.0  1.0  1.0  1.0  1.0  1.0  1.0  .67  .25  .33
ramp_v   (5→40°)      Ta1  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  .33  .75
cross_slope (5→30°)   Ta1  1.0  1.0  1.0  1.0  1.0  1.0  .83  .83  1.0  .71
round_log (Ø.15→.40)  Ta1  .50  .60  .86  .67  1.0  1.0  1.0  1.0  1.0  1.0
single step one-way   Ta1  ascending (even cols) ~1.0→.6, descending (odd) ~.5–.6 beyond 0.25
```

Readings:
- **Step-up / climb-out cliff at 0.30 → 0.333 m, and it is geometric.** With the front pair at
  the config's 60° up-limit the belt's underside at the tip sits at
  0.1165 + 0.3035·sin 60° − 0.078 = **0.30 m**; a higher edge cannot be hooked. Raising
  `marv_flipper_front_up_deg` to 80° would move it to ≈ 0.34 (90° → 0.34, the real joint is
  continuous). Under the current config, 0.30 is the ceiling — the v2 top columns are set there.
- **Stairs fall off between 0.10 and 0.16 rise on a 0.28 tread** while the same policies do
  0.15 rise on custom_mixed's 0.30 tread at ≥ 0.95: tread overfitting. v2 mixes treads
  (0.26 / 0.28 / 0.30, holdout 0.27 / 0.32) at 0.10 → 0.17 rise.
- Bridging gap ≤ 0.45 (0 at 0.50) — v2 range kept. A/V ramps to ~30°, side slope to 30° —
  v2 ramps 8 → 30°, cross-slope raised to 8 → 25°. Drops beyond ~0.25 m succeed ~50 %.
- Small round logs (Ø 0.15–0.18) were *harder* than big ones for the diffusion policy — the
  belly straddles them; kept Ø 0.15 → 0.28.

The pre-regrade `mixed_v2_straight` scored 0.855 for `diff_p1_Ta1` (2 episodes/spot), with the
losses exactly where the calibration predicted (stairs cols 6+, step-up cols 8–9, wave,
diagonal_pyramid, twin_rails); the re-graded course is what is now generated.

## 7. Verification done
- `pytest src/FTR-Bench-terrain-gen/tests` — 72 tests (painters vs analytic planes, mesh
  vertices == heightmap, adjacency rules, mirror symmetry, birth passes, offsets, stump field).
- `xformOp:orient` = `[1, 0, 0, 0]` on all four courses; `env_type_registry` resolves all four;
  every `mixed_v2_full` spawn maps to its own tile (6 per tile).
- Container evals on all four courses (this is also what exposed the UTF-8 `read_text` bug that
  had every *local* eval fall back to generic `env_NN` names).
- Container smoke training on `mixed_v2_straight` (32 robots, 2 batches): collection, update,
  per-batch eval, 32 env types in `eval_per_env.csv`, `steps_total` in `eval_per_spot.csv`,
  checkpoints. (The host-side `scripts/train.sh` crashes on this laptop at the first PhysX step
  on *any* terrain, custom_mixed included — an Isaac install issue, not the courses.)

## 8. Course extras added while watching the GUI eval
- Birth list in whole-course **passes** (all tiles forward, all reverse, then the jittered
  starts) so N robots land on N different tiles.
- **Out-of-bounds = the spot**: `CrossingEnvCfg.out_of_range_shape: rectangle`
  (start..target + 1 m along the path, ±1.67 m across); set in `configs/terrain_v2/*`, old runs
  keep the ellipse.
- **Arena band**: flat 2 m ring + 1 m fence around every course, outside all bounds.
- Turning rows: 1.0 m gate walls, 1.2 m lane side walls.
- `stump` graded 0.20 → 0.35; new `stump_field` (3 overlapping staggered bumps, growing along
  the path). The layout is randomised per tile (`jitter`: ±0.12 m along the path, 0.30–0.45 m lateral offset, ±6 % width, −10 % height on all but the tallest bump) so no two repeats share a flank sequence.
- Holdout: `feature_offset: {x: 0.5, y: 0.4}` — every feature sits at a seeded random offset
  in its tile (the training courses are always centred).
- Visuals, eval-only: `usd/<name>_decor.usd` (red tile lines, green start discs, blue goal
  squares with black/yellow hazard frames, red start-to-start lines; obstacles white, ground
  grey) loaded with `terrain_decor: true` — the eval wrapper sets it for non-headless runs.
  `sky_color` / `dome_light_intensity` / `sun_intensity` on `FtrEnvCfg` for the GUI sky.

## 9. Hills, friction, skid-steer and the flipper lock (2026-09-20)

### The stability rows
`steep_hill` (16 → 30°, 1.4 → 1.8 m ramps: both grade), `twisted_hill` (14 → 28°, roll ±6–14°
changing sign 1.5× along the path), `bumpy_hill` (14 → 28°, 30 seeded random bumps/hollows,
Gaussian about the drive line σ 0.6 m, clipped to ±1.45 m so they cannot be driven around).
All three ramps are built as **2 cm stairs with true vertical risers** (`step_tread: 0.02`,
`usd_utils.add_surface_mesh` on a non-uniform vertex grid). Hand-driven verdicts: 38° was too
much, 30/28/28° right; smooth ramps were "just climbable" at any friction and *worse* at μ 6
than at 4 — the risers, not the coefficient, are what gives the wheels purchase.
`stump_field` is randomised per tile (`jitter`).

### Open-loop hill probe (`terrain_config_hill_probe.yaml`, Ta1 and fixed postures)
Smooth 1.8 m ramps 15 → 45°, one row each at friction default / 2 / 8. Flat flippers: stuck
0.35 m into the ramp from 35° up at every friction (geometry — the tips cannot pitch the
rigid 1.12 m track onto the slope), and stalled at the crest edge at 28–32° (hull, 7 cm
clearance, μ 0.1). Front pair 60° up: 38° at default friction, 45° only at μ_eff 32. Ta1
(never saw a hill): 35° regardless of friction.

### Friction vs. turning — the real numbers
Plain-ground friction had been wheel 4.0 × terrain 1.2 = **μ_eff 4.8** (combine mode multiply).
On our own flat tile, full pivot command, flat flippers:

| μ_eff | 4.8 | 2.4 | 1.2 | 0.6 |
|---|---|---|---|---|
| pivot yaw rate | ~0.03 rad/s | 0.06 | 0.11 | 0.34 (20°/s) |

PhysX's (GPU-only) patch friction locks the yaw at high μ; Coulomb theory says the ratio
should not depend on μ, the sim says it does. The FTR `ground` terrain is a *dynamic* rigid body
with an articulation root (a loose plate) — never use it for turning numbers. Two more things
were wrong on top: the w action was only clamped (w ≤ 1 rad/s) and `set_v_w` used L = 0.36
(FTR body width) instead of MARV's 0.496 m gauge, so a full pivot ran the tracks at ±0.18 m/s.
New `FtrEnvCfg` knobs: `track_gauge`, `track_ang_vel_max` (now a scale on the action),
`track_vel_limit` (per-track saturation; the forward command keeps its own cap), all default
to the legacy behaviour. The v2 configs: gauge 0.496, w_max 4.0 → pivot at ±1.0 m/s.
Teleop's stick was mapped backwards (stick right = +w = CCW = left); the env's convention is
ROS-style and unchanged.

### Where the grip comes from instead: edge lips
The real track is hard rubber with ~1 cm protrusions: not grippy on smooth ground, but it
catches step *edges*. `edges.py` puts a **nosing** along every rising edge of a tile's
heightmap (≥ 4 cm): 3 cm overhang beyond the riser face (so no face is coplanar with the
slab's — a wheel meets one material at a time), 1 cm proud of the tread, 3 cm down the face,
own material (`edge_lip: true` or `{friction, height, width, min_rise}`), orange in the GUI,
not in the observation map. Rows: platform, pit, trenches, rails, beam, half platform, all
stairs, cobbles, rock, stepfield. Round/rotated rows (logs, trunk, pyramid, stumps) keep a
whole-row material (2–3); slopes 2; the hills 1.5 on the treads with the risers as their own
mesh prim at the lip friction (`riser_path`), so they can still be turned on.
Round logs get grip **ribs** instead (`RoundLog.build_lips`: tangent slabs every 3 cm of arc
over the upper half, 1 cm proud, at the lip friction). Plain ground in `configs/terrain_v2/*`:
wheel 1.0 × terrain 0.8 = **0.8**, hull 0.02.

Ta1 (trained at μ 4.8, so every drop is partly the policy) on the straight course, success
rate: old physics **0.90**; plain 0.8 without lips 0.76; overhanging lips μ 2 / 3 / 4:
0.78 / 0.80 / 0.75; lips μ 3 + hull 0.02: **0.83** (1.5 cm nose overhang instead of 3: 0.77).
The first lip build (flush with the riser face, μ 8) sent every lipped row to 0 % — wheels
jammed against two coplanar materials. On `marv_calib` the capability ceilings are the same
under old and new physics (step-up 0.30, deep gap, pit), ramps reach a column further, stairs
lose their easy columns — the remaining gap to close with a policy trained on the new physics
(`lowered_stairs`, `twin_rails`, `wave`, `tilted_pallet`). Tables: `logs/friction_eval/`.

With the final physics (5/2 iterations, plain 0.8, new drive) on the flat tile: pivot 17°/s
with flat flippers, 21°/s with both pairs raised; a one-track turn while driving (v 0.5,
w 0.5 → tracks 1.0 / 0.0) only 2°/s — turning while moving needs a large w (counter-rotating
tracks), which the policy is free to output.

### The flipper lock
"The flipper cannot be raised while the tracks drive" — measured: pushed against a 0.27 m
step with the tracks driving, the front flippers lift at ~10°/s with the joint drive pinned
at its 1000 N m limit; with the tracks stopped they lift at 90°/s; even the *rear* flippers
(on flat ground behind) are slowed. Not friction, not the lips, not wheel-drive reaction
torque (capping the wheel drives at 10 N m changes nothing), not the flipper-rotation wheel
correction (off/flipped: nothing). It is the **articulation solver not converging** with 20
wheel drives + contacts + a 3e4-stiffness joint drive at the Optuna-tuned 5 position / 1
velocity iterations. Time for the loaded flippers to lift 45° (three step heights):

| pos/vel iterations | 5/1 | 5/2 | 8/2 | 12/2 | 16/4 |
|---|---|---|---|---|---|
| lift time | 2.75–3.25 s | 1.25–1.75 s | 0.75–2.25 s | 1.25–1.75 s | 0.75–1.25 s |

Velocity iterations 1 → 2 is the lever; higher position counts buy little and have exploded
PhysX in the past. The v2 configs and `teleop_v2.yaml` use **5 / 2**.

### Hand testing
`CONFIG=teleop/teleop_v2.yaml bash scripts/teleop.sh --spawn_row NAME --spawn_col N [--reverse]`
— free drive (no goal/rollover/out-of-bounds/timeout resets), Start = respawn, Back = next
spawn, robot-follow camera, the training config's env overrides applied, `env_cfg_overrides.K=V`
accepted on the command line. `--dry_run N --dry_run_action V,W,FL,FR,RL,RR
[--dry_run_switch S --dry_run_action2 ...] --headless` is the gamepad-free probe used for
every measurement above (prints pose, yaw rate, wheel surface speeds, flipper angles, joint
torques).

## Sources
- Jacoff et al., *Stepfield Pallets*, PerMIS 2008 — https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=824741
- RoboCupRescue rules 2025F / 2022 Mobility — https://rrl.robocup.org/rules/
- Číhala, Pecka, Svoboda, Zimmermann, ICRA 2025 — https://arxiv.org/abs/2503.14389
- Zhang et al., *FTR-Bench*, JFR 2025 — https://onlinelibrary.wiley.com/doi/10.1002/rob.22528
- Pan et al. 2023 — https://arxiv.org/html/2306.10352
- LP-ACRL 2026 — https://arxiv.org/html/2601.17428v1
