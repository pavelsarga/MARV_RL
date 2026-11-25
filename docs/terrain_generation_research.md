# Terrain and robot-parameter randomization for a generalizing MARV policy

*Written 2026-11-24. Survey of terrain-generation and parameter-randomization methods for UGV RL
(any platform: legged, wheeled, tracked), and the proposed design for MARV (tracked, four
articulated flippers, position-controlled flippers). Follows `docs/terrain_v2_design.md`;
all MARV numbers below come from it or from the configs.*

## 0. Summary

- **Problem.** Every MARV course so far is static: a few dozen hand-written obstacle classes ×
  a 1-D difficulty grade, built once into one USD. The policy memorises it, and each shortcut has
  been closed by hand (§1). The robot is fixed as well: one set of flipper limits, one flipper
  speed, one drive speed and one friction/mass. A position-control policy can memorise that
  embodiment too.
- **Literature verdict.**
  - Fixed terrain sets overfit, whether a human or an LLM writes them (Eurekaverse).
  - Generalization grows with the *number of distinct* training levels (Procgen).
  - Adaptive selection and frontier-level generation (PLR, ACCEL, LP-ACRL, Verti-Selector) beat
    uniform sampling.
  - Much of the measured benefit of fancy curricula is simply an easy start (UED quadruped
    benchmark).
  - Nobody has published terrain generation or embodiment randomization for flipper-tracked
    robots.
- **Proposal.**
  1. A worker regenerates the course *during* training with the full generator, guided by the
     policy's per-tile statistics.
  2. The sim reloads the new course: about 1 min per reload; generating a 310-tile course takes
     40 s.
  3. The period is adaptive, driven by a per-tile visit budget (about 24–32 episodes per tile
     instance).
  4. Robot and physics parameters are randomized per episode with ADR.
  5. One *optimistic*, parameter-conditioned feasibility model ties the two together.
- **Baseline network.** Nothing changes in the architecture. Observing known parameters and a
  privileged critic are ablations only.

## 1. Why the current courses overfit

Symptoms already measured on MARV (terrain_v2 §1, §6; `variants.py` docstring; mixed_v3 header):

| shortcut the policy learned | how it was patched |
|---|---|
| handedness: diagonal_trunk 0.97 at −45° vs 0.31 at +45° | `mirror_alternate` |
| stair tread: 0.15 rise on a 0.30 tread ≥ 0.95, falls off by 0.16 rise on a 0.28 tread | mixed treads |
| layout lottery: cobblestones 0.39 at col 7, 0.71 at col 8 (seed, not difficulty) | neighbour rules |
| obstacle at a fixed distance from spawn (sd 0.02 m) | `feature_offset` |
| every drop follows a climb of the same height at a fixed distance; first edge goes up 73 % of the time | `lowered`, `extra.sample` |

These are all one failure mode, **parameter-space memorization**: the policy learns the
generator's correlations, not the physics. A generic list of causes (for any UGV):

1. **Diversity in parameters ≠ diversity in behaviour.** A parametric family can be wide in its
   knobs and still narrow in the situations it produces.
2. **One difficulty axis** orders a multi-dimensional space (height × length × asymmetry ×
   slope × roughness), so whole regions are never sampled.
3. **Infeasible samples.** For a robot with hard geometric limits (MARV: 0.30 m edge, ~0.45 m
   gap), naive randomization produces impossible tiles that teach giving up.
4. **Uniform replay of a fixed set.** Mastered tiles keep consuming frames, impossible ones give
   no signal, and a timed-out tile costs 4–6× the frames of a solved one (terrain_v2 §4).
5. **Evaluating on the training distribution.** A new seed of the same generator measures
   interpolation, not generalization.

## 2. State of the art — terrain generation

### A. Parametric sub-terrain grid + game-inspired curriculum (de-facto baseline)
A grid of sub-terrains (slopes, rough slopes, stairs, discrete boxes, gaps): rows are difficulty,
columns are type. A robot is promoted after success and demoted after failure.

- legged_gym / Rudin et al. 2022 ([2109.11978](https://arxiv.org/abs/2109.11978)).
- Isaac Lab `TerrainGenerator` + `TerrainImporter.update_terrain_levels`
  ([docs](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.terrains.html)).
- Followed by CTS ([2405.10830](https://arxiv.org/abs/2405.10830)), Extreme Parkour
  ([2309.14341](https://arxiv.org/abs/2309.14341)) and most quadruped work.
- *For:* controllable, cheap, crisp edges, and a built-in easy start. *Against:* this is MARV's
  current design and has failure modes 1–2.

### B. Procedural noise / natural-terrain synthesis
Perlin/simplex fBm, ridged multifractal, domain warping, Worley cells and erosion. Used for
"rough" sub-terrains and in off-road simulators: VW-Chrono
([2409.02383](https://arxiv.org/abs/2409.02383)), OffTerSim/WROOM
([2404.08855](https://arxiv.org/abs/2404.08855)).

- *For:* unlimited variety; hills, swales and ravines. *Against:* band-limited, so it has **no
  vertical edge to hook**, which is the feature flippers exist for, and difficulty control is
  poor. Only useful as a base layer.

### C. Compositional, grammar- and program-based generation
- **Wave Function Collapse tiling** ([2403.00187](https://arxiv.org/abs/2403.00187)): 1570 mesh
  tiles (steps, stairs, ramps, rough ground) assembled under adjacency rules, then boxes added.
  This gives a combinatorial number of layouts that are valid by construction.
- **Eurekaverse** (CoRL 2024, [2411.01775](https://arxiv.org/abs/2411.01775),
  [project](https://eureka-research.github.io/eurekaverse)): an LLM writes terrain programs,
  policies train on them, and the best results feed the next round of programs.
  - **Fixed terrain sets, human- or LLM-designed, overfit; the evolving curriculum keeps
    improving without plateauing.** It beat human-designed terrain on a 20-course held-out
    benchmark (about 2 more goals per course).
  - Generation happens **per round, i.e. periodically**, with a validity check that about half of
    the generated programs pass.
- *For:* keeps structure (edges, stairs, sequences) while multiplying combinations. MARV's
  `sequence` row is a one-rule instance of it. *Against:* needs a feasibility guarantee.

### D. Real-world terrain (scans, DEMs)
- **GaussGym** ([2510.15352](https://arxiv.org/abs/2510.15352)): simulation environments from
  phone, SLAM or 3D-dataset scans.
- **TTT-Parkour** ([2602.02331](https://arxiv.org/abs/2602.02331)): procedural pre-training, then
  fine-tuning on reconstructed real terrain in under 10 min.
- **VW-Chrono**: elevation maps of a physical testbed.
- **PGTT** ([2510.18348](https://arxiv.org/abs/2510.18348)): median-fills scan holes.
- *For:* true real-world statistics. *Against:* few samples, artifacts, no difficulty control.
  Best used as a holdout or a fine-tuning set.

### E. Generative models of terrain
- **ADTG** (CoRL 2024, [2410.10766](https://arxiv.org/abs/2410.10766)): a DDPM trained on
  natural terrain. It steers generation by optimizing the initial noise as a blend of noised
  training terrains *weighted by the current policy's performance on each*, so new terrains are
  novel and at the right difficulty.
- **ADEPT** ([2506.01759](https://arxiv.org/abs/2506.01759)): extends ADTG to multi-layer maps
  and teacher→student zero-shot sim-to-real. It reports beating both procedural and natural
  terrain sets.
- **ADD** (NeurIPS 2024, [2410.19715](https://arxiv.org/abs/2410.19715)): a diffusion model as
  the environment generator, steered by regret.
- **GenTe** ([2504.09997](https://arxiv.org/abs/2504.09997)): generative real-world-style
  terrains, used for evaluation.
- *For:* natural relief, adaptive. *Against:* needs a corpus, and crisp hookable edges and
  feasibility are not guaranteed. Suited to the natural (hill/ravine/rubble) layer, not to
  steps.

### F. Adaptive selection over a pool (automatic curricula)
- **PLR** ([2010.03934](https://arxiv.org/abs/2010.03934)): replays levels in proportion to a
  regret proxy (positive value loss), plus a staleness term.
- **Verti-Selector** ([2409.17469](https://arxiv.org/abs/2409.17469)), *wheeled UGV on
  vertically challenging terrain*: samples terrain by TD error on revisit, plus staleness.
  Reports +23 % success and real-world transfer.
- **LP-ACRL** (ETH, [2601.17428](https://arxiv.org/abs/2601.17428)): online learning-progress
  estimate from episodic reward reweights task sampling, with no difficulty ordering needed.
  ANYmal D at 2.5 m/s on stairs, slopes and gravel.
- **TACL** ([2608.16164](https://arxiv.org/abs/2608.16164))†: learns a policy-conditioned
  difficulty function over trajectories on *unstructured* maps that have no explicit difficulty
  order. Reports +56 % over no curriculum.

### G. Adversarial, open-ended and quality-diversity generation
- **PAIRED** ([2012.02096](https://arxiv.org/abs/2012.02096)): generator rewarded by
  protagonist–antagonist regret.
- **ACCEL** ([2203.01302](https://arxiv.org/abs/2203.01302)): PLR plus *small edits* to
  high-regret levels, so complexity grows exactly at the frontier.
- **POET** ([1901.01753](https://arxiv.org/abs/1901.01753)): co-evolving environment–agent pairs.
- Adversarial quadruped terrain: Tang et al. 2020
  ([2008.00603](https://arxiv.org/abs/2008.00603)).
- **Quality-diversity terrains:**
  - GECCO 2022 ([2203.15172](https://arxiv.org/abs/2203.15172)): MAP-Elites archive of terrains
    for curriculum RL.
  - ICRA 2025 ([2506.01362](https://arxiv.org/abs/2506.01362)): terrains archived by
    **behaviour descriptors** (each failure penalty's share of the total), so the set covers
    distinct *ways of failing*, not distinct parameters.
  - Environment/agent MAP-Elites ([2305.01153](https://arxiv.org/abs/2305.01153)).
- **FCTS** ([2610.22619](https://arxiv.org/abs/2610.22619))†: tree search over generated terrain,
  expanded from *localized policy failures*. Largest gains on the hardest routes against Extreme
  Parkour and Eurekaverse.
- **Reality check, the UED quadruped benchmark** ([2610.40364](https://arxiv.org/abs/2610.40364))†:
  - 14 curricula × 5 quadrupeds under one protocol, and **no method wins everywhere**.
  - **Easy starts explain much of the gain** (Terrain-CL is in the top group 13/20 times,
    ACCEL 11/20).
  - A third-party reading reports that the random-level winners can lose on held-out structured
    terrain.
  - Take-away: keep an easy start, and evaluate on a structurally different holdout.

### H. The scaling argument for regeneration
Procgen ([1912.01588](https://arxiv.org/abs/1912.01588)) and CoinRun
([1812.02341](https://arxiv.org/abs/1812.02341)): the train–test gap closes as the **number of
distinct training levels** grows. Agents trained on a few hundred levels overfit; thousands
generalize. MARV trains on 310 tiles. This is the most direct argument for regenerating rather
than cycling a fixed course.

### Changing terrain inside a running Isaac sim
Not supported.
- Isaac Lab has no runtime terrain regeneration
  ([IsaacLab discussion #2824](https://github.com/isaac-sim/IsaacLab/discussions/2824)).
- GPU-PhysX triangle-mesh colliders are baked at initialization, and no tensor API moves or
  changes them
  ([NVIDIA forum](https://forums.developer.nvidia.com/t/change-pose-of-triangle-mesh-collider-during-gpu-physx-simulation/352805)).
- Heightfield `modifySamples()` would need custom PhysX integration.
- The only in-process option is re-posing pre-spawned kinematic bodies, which restricts geometry
  to a fixed kit.
- **Reloading the sim with a freshly generated course** keeps the full generator, and on MARV it
  is cheap (§5.1).
- **Update (2026-11-25): swapping the whole course prim in-process works** for a static mesh
  course. Removing `/World/terrain` and referencing the new USD at the same path is picked up by
  PhysX on the next step, because no articulation or tensor view references the terrain. Measured
  with 8 MARVs, 6 swaps between `mixed_v3_holdout` and `mixed_v3_straight`: 2.5–3.5 s per swap,
  robots settle at the new course's height within 1 mm, GPU memory flat after the first swap.
  The limitation in the threads above concerns *moving or deforming* a baked collider, not
  replacing it. The relaunch is kept only as a fallback (§10).

## 3. State of the art — robot and physics parameter randomization

- **Fixed-range DR** (Tobin 2017 [1703.06907](https://arxiv.org/abs/1703.06907), Peng 2018
  [1710.06537](https://arxiv.org/abs/1710.06537)): ranges too wide make the policy conservative,
  too narrow make it brittle, and choosing them is hard
  ([2209.11596](https://arxiv.org/abs/2209.11596)).
- **ADR** (OpenAI, [1910.07113](https://arxiv.org/abs/1910.07113)): each parameter's range
  widens when performance at its boundary exceeds a threshold and narrows otherwise. Extended to
  locomotion together with reward-penalty scaling (TU Darmstadt,
  [report](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/OlegArenz/Leon_Magnus_Expert_Lab.pdf)).
  Learned DR distributions: Mozifian et al. ([1906.00410](https://arxiv.org/abs/1906.00410)).
- **Hidden parameters** (friction, mass, latency), handled with history or privileged
  information:
  - RMA ([2107.04034](https://arxiv.org/abs/2107.04034)): a teacher with privileged parameters,
    and a student that estimates a latent from proprioceptive history.
  - ROA / CTS ([2405.10830](https://arxiv.org/abs/2405.10830)): concurrent teacher–student.
  - Asymmetric actor–critic: only the critic sees the true parameters.
  - Known weakness: the student lags the teacher because of the information gap.
- **Known parameters** (configured limits, morphology), handled by conditioning on a description
  vector:
  - Embodiment randomization over 50 legged robots with a performance curriculum (Bohlinger &
    Peters, [GRC 2026](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/NicoBohlinger/grc_2026.pdf)).
  - McARL ([2505.18418](https://arxiv.org/abs/2505.18418)): morphology vector to actor and
    critic, zero-shot Go1→Go2.
  - GenLoco ([2209.05309](https://arxiv.org/abs/2209.05309)).
  - Embodiment scaling laws ([2505.05753](https://arxiv.org/abs/2505.05753)): zero-shot transfer
    to real robots with **modified joint constraints**.
- **Practical notes from the literature:**
  - Motor max velocity and torque are the hardest embodiment parameters to identify, which is an
    argument for randomizing them.
  - Inertia must stay physically consistent, or the simulation destabilizes.
  - Static friction is usually missing from DR and matters for sim-to-real
    ([2503.01255](https://arxiv.org/abs/2503.01255)).
- **Tracked/flipper robots.**
  - AT-D3QN/ICM-D3QN ([2306.10352](https://arxiv.org/abs/2306.10352)) and FTR-Bench
    ([JFR 2025](https://onlinelibrary.wiley.com/doi/10.1002/rob.22528)) randomize terrain noise
    and perception only.
  - **No published work randomizes flipper limits or speeds, or couples terrain generation to
    the robot's embodiment.**

† Preprint read through its abstract and secondary summaries only; the numbers are the authors'.

## 4. Robot and physics parameter randomization for MARV

### 4.1 What to randomize
Per episode at reset, through the PhysX tensor API, as Isaac Lab's `mdp.events` do:
`set_dof_limits`, `set_dof_max_velocities`, `set_dof_stiffnesses` / `set_dof_dampings`,
`set_dof_max_forces`, `set_masses` / `set_coms`, `set_material_properties`. Independent of
terrain reloads.

| group | parameter | nominal (config) | proposed range | why |
|---|---|---|---|---|
| flipper | up-limit, front / rear | 60° (`marv_flipper_*_up_deg`) | 45–80° | sets the step-up ceiling (§6) |
| flipper | down-limit, front / rear | 80° | 60–90° | belly clearance, tiptoe posture |
| flipper | max angular speed (joint `velocity_limit`) | 180°/s | 30–180°/s | real flipper motors are far slower than the sim default; timing at edges |
| flipper | drive stiffness / damping | 3e4 / 1000 | ×0.3–×2 | position-tracking lag |
| flipper | effort limit | 1000 N·m | ×0.3–×1 | loaded lift ("flipper lock", terrain_v2 §9) |
| flipper | encoder offset, backlash | 0 | ±3°, 0–2° | calibration error |
| tracks | max speed `track_vel_max` | 0.7 m/s | 0.4–0.9 m/s | approach momentum, gap bridging |
| tracks | left/right gain mismatch | 1.0 | ±5–8 % | heading drift; realistic and uncorrectable at ~4 % yaw authority |
| tracks | acceleration limit, motor effort | — | sampled | |
| body | mass, CoM x / z | ≈ 50 kg | ±15 %, ±5 / ±3 cm | payload; slope and tip-over limits |
| contact | wheel / flipper / hull μ; terrain μ static and dynamic; lip μ | per config | log-uniform around nominal | track grip, a dominant unknown |
| timing | action and observation latency | 0 | 0–100 ms | real control loop |
| sensing | heightmap noise, holes, drift (`heightmap_emulation`); IMU and encoder noise | — | widened | real elevation mapping |

**Not randomized: solver iterations.** They are numerics, not physics; changing them brings back
the articulation-solver flipper lock (terrain_v2 §9).

Implementation note: friction is currently baked into USD materials
(`assets/articulation/marv.py:141-155`) and shared by all robots. Per-robot friction has to go
through the tensor API (material buckets, which work with `replicate_physics=True`).

### 4.2 Schedule
ADR per parameter, starting at the nominal point. A fraction of episodes runs one parameter at a
range boundary. A boundary widens at success ≥ 0.7 and narrows at ≤ 0.4, with the success
measured relative to the same tiles at nominal.

### 4.3 Policy interface: no architecture change in the baseline
- **Fixed absolute angle mapping** for the position action, clipped at the drawn robot's limits.
  The same action always means the same angle. The policy sees saturation in the flipper-angle
  observations already in its input and history stack (T_o frames). The same history lets it
  partly infer hidden parameters (lag, friction, latency).
- **Ablation E3a:** add the configuration-known values (limits, max speeds) as a few extra
  observation dimensions. Input size changes, nothing else, and it is deployable because the
  real robot knows its configuration.
- **Ablation E3b:** a privileged critic (true hidden parameters to the value network only). Its
  only purpose is variance reduction: with randomized unobserved parameters, the same observation
  leads to different returns, and a critic without them predicts the average, so advantages get
  noisier. Training-only, but it touches every trainer. Used only if E2 shows noisy value
  estimates under wide DR.

## 5. Terrain that changes during training: periodic regeneration + reload

### 5.1 Mechanism
- **Generator worker**, in parallel with training. It runs the full existing generator
  (`MeshObstacle` surfaces, edge lips, hills with risers, `lowered`, `extra.sample`,
  `feature_offset`), so any geometry can be produced to order. It writes the next course under a
  new terrain name (`usd/`, `map/`, `birth/`, `gen_config/`), plus a per-tile `tiles.json` with
  descriptors and normalized difficulty (§6).
- **Generator input:** a statistics snapshot the trainer writes: per-tile visits and success, the
  success model (§6), current ADR ranges and descriptor-archive coverage.
- **Course composition:**
  - ≈ 30 % **mutations of frontier tiles** (ACCEL): new instances at the same or a harder
    difficulty, never copies.
  - ≈ 50 % **fresh samples aimed at under-covered descriptor cells** (a MAP-Elites archive over
    3–4 descriptors: max rising edge, max drop, gap width, pitch/roll grade, asymmetry, roughness
    band energy).
  - ≈ 20 % **uniform fresh samples**, easy ones included (easy start, anti-forgetting).
- **Tile grammar:** `Approach (Feature Gap){1..3} Exit` over the existing obstacle classes.
  - Every parameter is continuous, and the gaps are sampled, so approach speed and posture are
    decisions.
  - Features sit on low-frequency macro relief (≤ 12–15° along, ≤ 10° across), not flat ground.
  - Surface rubble and rocks, plus randomized bevels and lips.
  - Mirror, invert, reverse, small yaw (≤ ±10°, because yaw authority is ~4 %) and lateral
    offset.
  - Every tile is generated in the un-rotated frame (orient `[1, 0, 0, 0]`).
- **Reload:**
  - At an iteration boundary the trainer saves its training checkpoint and exits with a dedicated
    code (e.g. 76).
  - The launcher relaunches with `terrain=<next>` through the existing resume path
    (`_restore_training_state`: optimizer, schedules, cumulative frames; the persisted replay
    buffer for SAC, whose transitions stay physically valid after a terrain change).
  - The relaunch-and-account logic of `slurm/lib/respawn_common.sh` (exit code → relaunch,
    frame budget, schedule pinning) is reused with a terrain-rotation branch, plus the same loop
    in `scripts/train.sh` for local runs.
  - Per-row breakdowns keep working within a course (each has its own gen_config). Curves across
    courses use the descriptor and difficulty bins in `tiles.json`.
- **Measured cost:**

| item | value | source |
|---|---|---|
| reload (launch → env, terrain, checkpoint restored), 1024 robots, 45 MB course | **≈ 1 min** (launch 13:22:26, Kit ready +16 s, resumed 13:23:27) | `logs/diff_p1_smooth_ramp0_11826795/attempt_9` |
| training throughput, diffusion PPO, 1024 robots | ≈ 290 frames/s (2.1 M frames in 2 h) | same, weight timestamps |
| generate a full `mixed_v3_straight` course (310 tiles, 1860 spawns, 19 k edge lips) | **39 s** wall-clock, 335 % CPU, 0.55 GB peak RAM, 75 MB on disk | measured 2026-11-24 |

So a 2 M-frame period costs < 1 % in reload overhead, and even 0.5 M frames ≈ 3.5 %. Generation
is about 1/150 of a period, so one worker is enough. Run it at low priority, because its
multi-core burst competes with the simulation's CPU-side work (heightmap lookups).

### 5.2 How often to change terrain
**Recommended: an adaptive period driven by a per-tile visit budget.**

- **Within a course**, the spawn sampler replaces `cycle()` in `FtrEnv._prepare_reset_info`
  (`ftr_env.py:947-967`) and works as follows:
  - **Visit budget:** at most N_max ≈ 24–32 episodes per tile instance. That is about the number
    of episodes needed to estimate a tile's success rate to ±0.1, and it is the anti-memorization
    guarantee.
  - **Mastered:** a tile is retired early once its Beta posterior gives P(p > 0.85) > 0.9.
  - **Dead:** a tile is down-weighted when it shows no learning progress after N_dead episodes.
    If it was a probe tile, that is recorded for the feasibility model (§6).
  - Among the remaining tiles: p ∝ ε + learning progress + β·staleness (LP-ACRL, Verti-Selector),
    with an easy-start prior over normalized difficulty early in training.
- **Reload trigger:**
  - Reload when **≥ 75 % of the tiles are exhausted, mastered or dead**, bounded by a minimum
    and maximum period (e.g. 0.5–4 M frames).
  - The worker starts generating the next course at ≈ 60 % from a snapshot, so the course is
    ready before the trigger fires.
- **Why not the two simple options:**

| schedule | problem |
|---|---|
| fixed period, whole course | discards unlearned tiles early and keeps mastered tiles too long; the period is a guess; every reload is a full distribution jump |
| mastery-triggered only | tiles the policy never masters are driven hundreds of times and memorized (the Procgen effect); the trigger can wait indefinitely |
| **visit budget + mastery/dead exits + adaptive period** | the cap rules out memorization, the exits spend frames where learning happens, the period follows learning speed, the bounds keep it predictable; the ≈ 30 % mutated carry-over keeps frontier difficulty across a reload and softens the jump PPO's value function sees |

No published work isolates the change frequency as a variable. Eurekaverse regenerates per round
and PLR/ACCEL replace levels in a buffer by score, so the choice is tested directly in E4 (§8).

- **Scale.** A 300-tile course × 24 episodes ≈ 7 k episodes ≈ 1.5–2.5 M frames at 1024 robots,
  so a period lasts about 1.5–2.5 h at current throughput. 50 M frames is ≈ 25 courses ≈
  **7.5 k distinct tile instances** (against 310 today), for ≈ 25 min of total reload time.

## 6. Feasibility: optimistic and conditioned on the robot parameters

Whether a terrain is crossable depends on terrain and robot together, F(d, θ).

- **Capability prior c(θ)**, analytic and calibrated on `marv_calib` (terrain_v2 §6):
  - **Step-up / climb-out:** h_max(θ_up) = 0.1165 + 0.3035·sin θ_up − 0.078 = 0.0385 +
    0.3035·sin θ_up, i.e. 0.25 m at 45°, **0.30 m at 60°** (measured: 100 % at 0.30, 0 % at
    0.333), 0.32 m at 70°, 0.34 m at 80°.
  - **Gap:** ≈ 0.45 m (0 % at 0.50), lower at low max speed.
  - **Slope:** traction (tan s ≤ μ_eff) and the CoM tip-over line, capped by the hill probes
    (35–38° with the front pair raised).
  - **Drop, breakover:** drops above ≈ 0.25 m succeed about half the time; breakover against
    the 1.12 m track and 0.07 m belly (≈ 7° with flat flippers).
- **Normalized difficulty** r = max_k d_k / c_k(θ). One scalar ties terrain and embodiment
  together; curriculum statistics live in r-bins, so they carry across parameter draws and
  across courses.
- **Optimistic acceptance**, so that slightly larger obstacles stay in and challenge the policy:
  - accept r ≤ 1 + m, with m ≈ 0.15 (≈ +4–5 cm on a 0.30 m step)
  - keep **≈ 5–10 % probe tiles with r up to ≈ 1.4**
  - drop only r > 1.4
  - The prior is a prior. When probes succeed, the envelope is raised from data, which is how a
    policy that finds a better technique gets credit for it.
- **Learned residual:** an online success model p(success | d, θ), logistic in r plus
  descriptor features (TACL-style). It refines c(θ), and it is used twice:
  - by the spawn sampler when it draws θ for a chosen tile (rejection sampling inside the
    acceptance band, except for probes)
  - by the worker when it composes the next course for the current ADR ranges

## 7. Evaluation (static sets, never trained on)

- **Holdout sets:**
  - `mixed_v3_holdout` and `marv_calib`, as today
  - a **descriptor-split holdout**: whole descriptor regions withheld, e.g. 0.20–0.25 m drops on
    a roll-sloped base (extrapolation, not a new seed)
  - an **embodiment-split holdout**: up-limit 50° and 75°, flipper speed 40°/s, μ outside the
    training range
  - optionally, elevation patches from MARV LiDAR bags
- **Metrics:**
  - generalization gap (training-course success − holdout success)
  - success heatmaps by r-bin and descriptor bin
  - one-parameter-at-a-time sensitivity sweeps

## 8. Experiments

| # | experiment | question |
|---|---|---|
| E1 | sensitivity of the current best policies (Ta1, A0t) to each §4.1 parameter alone, on `marv_calib` + holdout | which parameters they are brittle to; sets the ADR order and ranges |
| E2 | no DR vs fixed-wide DR vs ADR, on static mixed_v3 | value of the DR schedule |
| E3a / E3b | + known parameters in the observation / + privileged critic | whether any interface change is worth it |
| E4 | static mixed_v3 vs regenerated courses; N_max ∈ {8, 24, 96}; adaptive vs fixed period | change frequency, answered empirically |
| E5 | feasibility margin m ∈ {0, 0.15, 0.3}; probes on / off | value of optimism vs wasted samples |

## 9. Pre-build feasibility tests (2026-11-24, laptop, MARV, Ta1/A0t physics)

Test scripts were scratch only; numbers below are from their logs.

**G1: per-tile programmatic generation.**
- Setup: a `CourseGrid` subclass whose `iter_tiles()` yields an explicit tile list, i.e. any
  obstacle type with continuous parameters per tile. It runs through the *unmodified*
  `assembler.generate()`, because everything downstream iterates `iter_tiles()`; `birth.py`
  only reads row-level settings from `grid.rows`.
- Result: 300 tiles, 43 types, 1800 spawns.

| step | time |
|---|---|
| heightmap + spawns | **0.6 s** |
| full USD (meshes, lips, decor) | 35.5 s |
| descriptors for 300 tiles | 0.06 s |

- Over-generating candidates, filtering on descriptors and feasibility, and building USD only for
  accepted tiles is therefore cheap.
- Descriptors validated on mixed_v3_straight:
  - edge height = configured platform height (0.200 m)
  - cross-slope roll 8.0° → 25.1° (configured 8 → 25)
  - ridge-ramp pitch 8° → 30°
  - deep-gap width 0.20 → 0.45 m
- To fix when building: `edge_max` also counts gap walls (0.6 m on deep_gap). Feasibility must
  separate a *climbed* edge from a *bridged* gap.

**G2: cold load of a never-loaded course.** `gym.make` took 8.5 s cold vs 4.6 s warm, so PhysX
mesh cooking adds ≈ 4 s. The whole process took 29 s. Reload cost is not a constraint.

**R1: per-robot randomization through the PhysX tensor API** (8 robots, each with a different
value). All setters need **CPU tensors** (GPU tensors are rejected with "Incompatible device") and
cost 0.1–3 ms per call for 8 robots.

| parameter | API | measured effect |
|---|---|---|
| max flipper speed | `set_dof_max_velocities` | peak speed equals the setting exactly: 30/60/90/180 °/s |
| flipper up-limit (software clamp of the position target) | per-env `low/high` | reached 45.0/55.0/70.0/80.1° for 45/55/70/80° |
| flipper limit (hard) | `set_dof_limits` | reached 30.0/40.0/50.0/59.9° for 30/40/50/60° |
| drive stiffness/damping ×0.3–×2 | `set_dof_stiffnesses/dampings` | applied, but rise time unchanged (0.95 s): masked by the speed cap. Randomize lag through max speed + latency instead |
| effort limit | `set_dof_max_forces` | tiptoe body lift 8.5 cm at 1000/300/100 N·m, 2.8–3.9 cm at 30 N·m. Only the 30–300 N·m band matters |
| mass, CoM | `set_masses/coms` | read-back exact; heavier robots sit 3–4 cm lower (flipper drives sag). Note `base_link` is only 8.7 of 58.4 kg, so scale all links or add a payload |
| friction (all robot shapes) | `set_material_properties` | pivot yaw rate 0.76/0.44/0.19/0.07–0.21 rad/s at μ 0.1/0.25/0.5/1.0 |
| subset update mid-run (2 of 8 robots) | same setters with indices | only those robots changed; no NaN, no explosions |

Robot shapes can be classified for per-group friction by their nominal value: of the 55 shapes,
wheels sit at 4.0, flippers at 6.0 and the hull at 0.1.

**P1: terrain rotation by relaunch.**
- Setup: fake `SLURM_JOB_ID`, train_diffusion with the A0t+V-Gov config.
  - attempt_0: mixed_v3_straight
  - attempt_1: relaunched with `--terrain <fresh per-tile course>`
- What worked:
  - the respawn path loaded the previous attempt's config with `--terrain` applied on top
  - policy, vecnorm, optimizer and scheduler state resumed (iteration 2, 1536 frames)
  - training continued on the new course with checkpoints every iteration
  - mid-training eval and `eval_per_env.csv` used the new course's layout
- Frame budget: on relaunch, `total_frames` acts as frames still to run (1536 + 3072 → 4608), as
  `respawn_common.sh` expects.
- Both attempts were stopped by the RAM watchdog during the *final* eval, the memory peak on this
  laptop. Training itself was unaffected.

**Bug found, to fix before building.** On every resume `train_diffusion.py` re-applies the
*configured initial* LR over the restored decayed LR (`Resume: config lr … is 0.000503,
checkpoint had 0.000402 — using the config value`). With a chainable scheduler (`LinearLR`), each
reload discards the decay so far, and at ~25 reloads per run the LR would never decay. Fix: compare
the config LR with the checkpoint's `initial_lr` / `base_lrs`, and rescale only when the config was
actually changed.

## 10. Implementation (built 2026-11-24)

| piece | where | tests |
|---|---|---|
| per-robot domain randomization + ADR | `ftr_envs/tasks/crossing/randomization.py`, hooks in `ftr_env.py` (`_reset_idx`, `_pre_physics_step`, `_apply_action`, `get_flipper_pos`) | R2 (Isaac, 8 robots: limits 0.27° max error, encoder, speed cap, L/R gain, latency, mass, friction groups, subset reset, ADR expand/shrink + state) |
| per-robot friction | tensor-API material edits on shapes classified by nominal friction (no USD change needed) | R1, R2 |
| adaptive spawn sampler | `ftr_envs/tasks/crossing/spawn_sampler.py`, `FtrEnv._prepare_reset_info` / `_on_episodes_end` | `FTR-Benchmark/tests/test_spawn_sampler.py` (10), S1 (Isaac, 16 robots) |
| per-tile courses, descriptors, feasibility, relief, composer, rotation CLI | `ftr_terrain_gen/{grid.TileListGrid, variants.ReliefObstacle, descriptors, feasibility, compose, rotation}.py`; `assembler.generate(grid=…)` | `FTR-Bench-terrain-gen/tests/test_compose.py` (13; suite 120) |
| trainer side: background generation, exit 76, ADR state in `training_state.pth`, eval pause | `training/terrain_rotation.py`, `train_diffusion.py` | E2E-1/E2E-2 (Isaac, 16 robots, 3 and 5 courses) |
| launchers | `slurm/lib/train_common.sh` (exit 76 branch), `scripts/train_rotation.sh` | E2E-1/E2E-2 |
| config | `configs/terrain_rotation/marv_config_diffusion_rotation_dr.yaml` | E2E-1/E2E-2 |

Changes from the design above, from the tests:

- **Flipper effort is not randomized** (§4.1 table): the drive limit-cycles when its gains change
  (R3 sweeps). Slower / weaker flippers = max speed + latency.
- `track_vel_max` draws above ~0.7 m/s add nothing — measured speed saturates at 0.64–0.8 m/s.
- New tiles are visited first (strictly), then weighted; `dead_threshold` 0.10 (Wilson-hi of 0/16 is
  0.093, so 0.05 would need ~31 episodes).
- Gaps are separated from edges by a morphological closing with a 0.9 m flat element (§6 demands).
- **Difficulty targeting (added 2026-11-25).** In the first version only the 30 % mutant share followed
  the policy: a simulated 10-course chain with skill rising 0.55 -> 1.10 kept the mean tile r at
  0.79-0.85 and success drifted 0.20 -> 0.80. Now `rotation.plan_next` fits a logistic success model
  p(success | r) to the last course (`feasibility.fit_success_model`; kept from the previous course
  when there is too little data) and the coverage tiles are drawn inside the r band where it predicts
  30-80 % success (`feasibility.target_band`, `ComposeConfig.target_band`); the 20 % fresh share stays
  uniform. Same chain: mean r follows the policy 0.62 -> 0.97, success 0.54-0.75, every coverage tile
  in band. Costs: archive growth 152 -> 267 cells instead of 379, and probes drop to 4-5 per course
  until the band reaches the probe zone. `--no-targeting` restores the old behaviour.
- **Open: tall, steep tiles.** 12-16 % of composed tiles are taller than 0.6 m (27-58 over 1 m in 3000),
  mostly the hill classes at up to 38-40 degrees: there is no total-height demand, grade probes reach
  30 x 1.4 = 42 degrees, and t <= 1.2 extrapolates the hills. Proposed: total-height demand capped at
  0.6 m (gap depth excluded, no probes past it), grade probes capped near 34 degrees, t <= 1 for hills
  and stairs — pending a look at the true-scale courses.

Bugs found and fixed while building (both affect every relaunch, hence terrain rotation):

- LR re-applied at its initial value on every resume (`trainer_common.reconcile_resumed_lrs`).
- Iteration off-by-one after every resume in all four trainers: the first iteration repeated the
  previous attempt's frame label, losing one iteration of schedule progress and overshooting the
  budget by one per relaunch. E2E-2 after the fix: labels 512→6144 contiguous over 5 attempts,
  budget met exactly.

E2E-2 (laptop, 16 robots, 3×4-tile courses, visit budget 2, 2 s episodes): 1 rotation on
`max_frames`, 4 on exhaustion; every next course ready 2–3 s after its background generation
started; `keep: 3` cleanup; `spawn.csv` / `dr.csv` / `rotation.csv` logged. Not yet exercised in
Isaac: frontier mutants from real statistics (needs >= 4 finished episodes per tile, impossible at
budget 2 — covered by the pytest rotation test and by running the rotation parser on E2E-2's real
snapshot) and ADR boundary moves from real episodes (needs a long run).
