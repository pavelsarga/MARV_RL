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

## 6. Verification
- `pytest src/FTR-Bench-terrain-gen/tests` (69 tests: painters vs analytic planes, mesh
  vertices == heightmap, adjacency rules, mirror symmetry, birth augmentation counts).
- `xformOp:orient` is `[1, 0, 0, 0]` on all four courses (checked with the `Gf.Quatd` snippet
  from CLAUDE.md).
- `env_type_registry.get_terrain_layout()` resolves all four; all 2040 `mixed_v2_full` spawn
  entries map to their own tile (6 per tile).

## Sources
- Jacoff et al., *Stepfield Pallets*, PerMIS 2008 — https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=824741
- RoboCupRescue rules 2025F / 2022 Mobility — https://rrl.robocup.org/rules/
- Číhala, Pecka, Svoboda, Zimmermann, ICRA 2025 — https://arxiv.org/abs/2503.14389
- Zhang et al., *FTR-Bench*, JFR 2025 — https://onlinelibrary.wiley.com/doi/10.1002/rob.22528
- Pan et al. 2023 — https://arxiv.org/html/2306.10352
- LP-ACRL 2026 — https://arxiv.org/html/2601.17428v1
