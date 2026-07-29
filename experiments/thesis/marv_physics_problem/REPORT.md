# MARV physics investigation — findings and fixes

Working notes from the session investigating why MARV (unlike FTR) struggles with
slow/laboured movement, bouncing, and getting stuck while climbing in the IsaacLab sim.

## STATUS: PRIMARY OBJECTIVE MET

`./scripts/eval.sh experiments/marv_physics_problem/attempt_0/ --num_envs 64 --repeats 4`

```
eval/success_rate    0.1239   ← target was > 0.1 ✓
eval/explosion_rate  0.0000   ← zero explosions
eval/failure_rate    0.0192
state/lin_velocity   0.0839 m/s (mean)
state/lin_velocity_max 0.480 m/s
```

The two root-cause fixes that got here:
1. **`flipper_wheel` stiffness=0** (Finding #17) — eliminated phantom restoring torque
2. **`disable_flipper_arm_collision: true`** (Finding #21) — eliminated overlapping rigid-body contact instability at wheel1 pivot

## Tooling added

- `scripts/teleop.py` + `scripts/teleop.sh` + `configs/teleop_marv.yaml` /
  `configs/teleop_ftr.yaml` — config-driven gamepad teleop, forked from
  `src/FTR-Benchmark/scripts/ftr_algo/teleop.py`, with:
  - the task's default `episode_length_s=30` (decimation=5, sim_dt=0.005) truncates and
    respawns the env every 1200 steps, cutting off manual driving sessions — now
    overridden to a large value via config.
  - a per-step CSV logger (`logs/teleop_wheel1_<timestamp>.csv`) recording every wheel's
    joint velocity (with a configurable glitch threshold), actual linear/angular velocity,
    root position, roll/pitch, and flipper angles.

## Findings

**1. Bouncing/violent depenetration — root cause found and fixed.**
`teleop.py` was launching with `parse_env_cfg`'s raw registered defaults
(`sim.dt=0.0025`, per-articulation solver position/velocity iterations 16/4,
`max_depenetration_velocity=0.15`, straight from `FTR_SIM_CFG`/`MARV_CFG`). Every
train/eval script (`train_ftr.py`, `eval_ftr.py`, ...) explicitly overrides these from
the run's `config.yaml` — for MARV, `configs/best_long_config_marv.yaml` uses
`sim_dt=0.005`, `solver_position_iterations=5`, `solver_velocity_iterations=1`,
`max_depenetration_velocity≈0.1235` (Optuna-tuned specifically to minimize crashes).
Teleop was silently running a completely different, untuned solver regime. Fixed by
applying the same overrides in `teleop.py`. Confirmed by direct A/B test: violent
bounce/launch events (one threw the robot ~90° and nearly out of bounds) disappeared
after the fix.

**2. Wheel1/wheel5 velocity glitch — confirmed present on FTR too, not MARV-specific,
but more frequent and more severe on MARV.**
Logged wheel-joint velocities during matched teleop sessions on both robots
(same terrain, same driving):
- FTR: 1.2% of steps flagged (>10 rad/s negative spike), all isolated 1–2 step blips,
  max magnitude ~95 rad/s. Never visibly disrupted the crossing.
- MARV (pre solver-fix): 5.3% of steps flagged, with at least one escalating
  catch/release cluster peaking at 311 rad/s on a single wheel — consistent with the
  violent 90° spin event observed.
- After the solver fix: glitch rate ticked up slightly (7.1%) but peak severity dropped
  sharply (max 75.7 rad/s, no more 300+ events) — solver mismatch was driving the worst
  spikes; a baseline glitch tendency remains.

**Per-wheel breakdown (all 5 wheels per flipper, not just wheel1) shows the glitch is
bimodal, not wheel1-specific:** wheel1 (lead/big) and wheel5 (tip/small) both glitch far
more than wheels 2–4 (e.g. one run: wheel1=307, wheel5=452, wheels 2–4 combined=105).
The existing contact-offset fix in `MarvWheelArticulation.set_robot_env` only tightens
`contact_offset` on wheel1 — per this data it should likely also cover wheel5. **Not yet
implemented — flagged for a follow-up, not done in this session.**

**Glitch is a symmetric single-frame impulse, NOT a sustained reversal — most likely
collision-solver contact ringing, not traction loss.** Characterised the post-solver-fix
glitch precisely (the earlier `v < -10` flag was one-sided and misleadingly suggested a
negative/reversal velocity):
- *Sign is symmetric.* Of all `|v|>15` excursions: 320 positive / 321 negative (max
  +67.9, min −66.6). During forward driving it's a mild +159/−113 — nowhere near "the
  wheel drives backward." The wheel is not reversing relative to rest; it blips equally
  hard both ways.
- *Shape is a clean 1-frame impulse.* Of `|v|>20` excursions, 270 are clean single-step
  (both neighbour frames back at the normal ~4–5 rad/s) vs only 96 multi-step. Typical
  shape: `…, -1.9, 31.5, 29.4, -2.3, …` — normal, one/two garbage frames, back to normal.
- *Mechanism.* wheel1/wheel5 are the two ends of the flipper's contact patch, continually
  making/breaking contact with terrain mesh edges as the robot moves (wheels 2–4 sit in
  stable continuous contact → calm). Each make/break is a fresh PhysX contact-constraint
  event applying a one-substep corrective impulse; the light (0.73 kg) velocity-driven
  wheel over/undershoots for a frame and the drive (damping=100) snaps it back. The
  *symmetry* is the tell — a real snag/traction-loss would be *directional* (consistent
  sign opposing motion). Symmetric single-frame blips = contact ringing.
- *Implication.* A symmetric single-frame impulse imparts ≈0 *net* momentum over time, so
  it is largely cosmetic — it does NOT explain the steady "slow and laboured" feel. The
  pre-solver-fix 311 rad/s events were genuinely disruptive (and are gone). The remaining
  ±50–67 single-frame blips are noise on top of the real problem, which is the sustained
  directional stall (#4). Conclusion: the edge-wheel glitch is **not** the core problem;
  deprioritised in favour of the stall.

**3. Wheel1 collision-radius experiment — inconclusive/confounded.**
Tested shrinking only wheel1's collision-shape radius (kinematic radius for velocity
targets left unchanged) at 0.95 and 0.7 scale. Both eliminated glitches entirely, but
also caused wheel1 to lose ground contact altogether (wheel2's natural radius is larger
than 0.95× wheel1's), forcing the robot onto its flipper tips — confirmed by the
reporter. "Zero glitches" in that condition is at least partly trivial (no contact, no
glitch source). Does not cleanly answer whether wheel1's contact-geometry size affects
snag rate independent of losing contact. Parked rather than pursued further.

**4. Sustained low-speed stall on flat ground, distinct from the spike glitches.**
In the solver-fixed baseline run, found a 45 s segment with `cmd_v` pinned at ~0.95
(full forward) where wheel1 settles at ~2.5–3 rad/s instead of its own ~8.15 rad/s
free-spin target — a sustained partial stall, not a transient spike (mostly inside the
glitch threshold so untagged by the spike detector). Net displacement over the full
~102 s session was only ~2.5 m (~0.025 m/s average against 0.95 m/s commanded).
**Root cause not yet identified** — distinguishing genuine wheel slip from "wheels
turning fine but body just not translating" needs a follow-up run with the
position/velocity logging now in place.

**5. URDF/conversion pipeline audit — no leftover front-left-specific friction
mismatch found.**
Checked the `robot_rodeo_gym_ros2` xacro/source pipeline (colleagues had been testing
controllers/friction on the front-left flipper specifically) against the
`elevation_mapping` branch. Found one real, unrelated regression: commit `c054db2`
("full control restored + naming fix") reverted `flipper_fake_wheels`'s recursion back
to a pre-fix form that would only generate 1 wheel per flipper (not 5) if the URDF were
regenerated today — symmetric across all four flippers, not front-left-specific, and
latent (the currently-deployed `marv.usd` already has all 5 wheels correctly, built
before this regression landed, so it doesn't explain current sim behavior). Verified
mass, inertia, joint-drive params, contact offsets, and material bindings are
bit-for-bit identical across all four flippers in the deployed asset — no asymmetry
found there.

**6. Hypothesis tested and REJECTED: flipper-wedge contact geometry is NOT the cause of
the stall.**
Measured every flipper wheel's center height/radius from the USDs and found MARV's 5
tapered fake-wheels (0.1165→0.078 radius, centers all on the flipper centerline) have
bottoms sloped 7.2° — at flipper angle 0 only wheel1 bottoms out; wheels 2–5 hang
9.6–38.5 mm above flat ground. Hypothesized this 4-point (wheel1-only) contact was
starving traction and causing the stall. **Reporter falsified this directly: FTR is
independently configured with the same sloped/tapered flipper-track geometry, so it is
not MARV-specific; and angling MARV's flippers to bring the full track into contact
during a live stuck episode did not fix the stall.** Re-checked against this same log:
flippers angled >10° actually moved *worse* (vx mean 0.036, n=455) than near-flat <3°
(vx mean 0.129, n=107) — the opposite of what the contact-count theory predicts.
**Conclusion: contact-patch count is not the bottleneck — a sim should handle sparse
point contacts fine (cf. wheeled/legged robots), and the data agrees. Root cause of the
stall is still open.**

Other observations from this pass, still unexplained:
- Wheel1 spin while driving forward rarely reaches its own free-spin target (8.15
  rad/s): p50=4.2, p90=5.4, p99=8.4 rad/s — consistent with the wheel being loaded/
  resisted rather than freely slipping, but doesn't pin down by what.
- The single best moment in the whole log (vx=0.45 m/s, still only half of commanded)
  had FL/FR flippers raised to ~55-59° with RL/RR near flat — an asymmetric, not
  obviously deliberate-looking pose. Not yet clear if this is signal or noise (n is
  small at the top of the vx distribution).

**7. Effort-limit raise tested — REJECTED, and the real mechanism found: continuous
frame-to-frame contact chatter on the lead wheel(s) during the stall.**
`flipper_wheel` actuator (`MARV_CFG`) sets no `effort_limit` at all — confirmed in
IsaacLab's `actuator_base.py` this defaults to `+inf`, so wheel spin torque was never
capped; "raise the wheel torque limit" was moot, there wasn't one. The only finite limit
anywhere in `MARV_CFG` is `flipper_joint` (pivot) at `effort_limit=1000`. Raised to 5000
and re-ran the same climb+stall scenario with full torque logging. **Reverted back to
1000 after the negative result below** — `MARV_CFG` is the asset config used by every
MARV run (training/eval/teleop alike), not just this diagnostic, so an unhelpful change
there isn't left in place.

**Result: no effect.** `vx` mean/median (0.047/0.026) was statistically identical to the
pre-raise run (0.047/0.039) — confirmed directly by the reporter driving it ("no angle
or contact area size changed the speed"). Pivot torque demand did peak at ~1837 N·m
this run (which would have clipped under the old 1000 limit, so the limit was real and
being hit), but removing that clipping changed nothing about forward progress. Pivot
saturation is not the bottleneck.

**What the torque/velocity trace actually shows, frame by frame, during the flat-ground
stall:** FL1 (and RL1) do not settle near their commanded target (8.15 rad/s, the exact
free-spin speed for `v=0.95`) — they slam between **+8.15 and −8.15 on alternating
frames** (`-8.15, 3.28, -6.13, ..., 8.15, -8.15, 3.12, -8.15, ...`), with torque spikes
up to ±2000–7000 N·m on the same frames. 17% of all flat-forward steps have FL1 sitting
exactly at the ±8.15 limit, and of those, 80% are the *negative* extreme — not
symmetric, unlike the whole-log average in #2. This is the same contact-ringing
mechanism characterized in #2 (edge wheels making/breaking contact), but here it is
happening on **every frame**, continuously, rather than 1–7% of the time. Net body
displacement during this window was never literally zero (3.9 m over 118 s, ≈0.033 m/s
average, all 5 s buckets show some progress) — the small leftover after the +/− spikes
mostly cancel is consistent with "wheel mostly fighting itself, occasionally netting a
little real thrust" rather than smooth rolling.

This unifies #2 and #4: there is no separate "stall mechanism" distinct from the
edge-wheel glitch — the stall is what the same contact-chatter looks like when its
duty-cycle goes from occasional (1-7%) to continuous (effectively 100%) in a particular
contact configuration. Why this specific configuration drives the chatter rate so much
higher is still unknown.

**8. Side observation: `cur_mixed` terrain mesh wireframe looks badly malformed near
obstacle geometry — flagged, not yet investigated.** Reporter's Blender screenshots of
the obstacle region's wireframe show a chaotic web of long crossing diagonal edges over
what should be a regular grid, vs. clean rectangular tessellation on the flat ground
right next to it. Plausibly relevant to contact chatter (rolling over a mesh with
overlapping/degenerate triangles could itself cause competing contact-solver
resolutions at a single contact point) but not yet connected to the chatter evidence
above — could equally be irrelevant. Parked behind the solver-iteration retest below;
revisit if that doesn't explain the chatter.

**9. Solver velocity iterations retest — REJECTED, made things worse.**
Reran the same climb+stall scenario with `solver_velocity_iterations=4` (CLI override
via `teleop.sh`, not a persistent config change) instead of the Optuna-tuned 1.
**Result: worse on every metric the reporter could see directly** — the violent
depenetration bounce that the earlier solver-mismatch fix (#1) had eliminated came
back. Confirmed in the logs: max wheel velocity 53.5 and 92.6 rad/s across two runs,
both higher than the post-fix vel=1 baseline's 75.7 rad/s ceiling, and trending toward
the pre-fix 311 rad/s territory rather than away from it. Session appears to have
restarted mid-test (two separate short log files ~12s and ~68s, consistent with a crash
forcing a relaunch, though not directly captured). No code change was made (this was a
CLI-only test), so nothing to revert. This closes the solver-velocity-iterations
avenue from both directions now: the original sr-based sweep (8/16/32, early in the
investigation) showed no improvement, and this targeted retest at vel=4 actively made
the chatter/bounce worse. **The Optuna-tuned `solver_velocity_iterations=1` is not the
problem — it's already near-optimal for this failure mode too, not just the original
explosion-avoidance it was tuned for.**

**10. `cur_mixed` terrain investigation — two real bugs found and ruled out as the
cause; one strong new lead found.**

Reporter flagged the `cur_mixed` mesh wireframe as visually chaotic (Blender
screenshots). Investigated the generation pipeline (`ftr_envs/assets/terrain/terrain.py`
+ `cur_mixed.yaml` + `cur_mixed.usd` + `cur_mixed.map`):

- **Real bug found: quaternion-order mismatch in `terrain.py`'s `apply()`.**
  `cur_mixed.yaml`'s `prim_config` sets `xformOp:orient: [0, 0, 0, 1.0]` intending
  identity (the universal `[x,y,z,w]` convention). The code does
  `Quatd(*value)` — but `pxr.Gf.Quatd`'s constructor takes the **real (w) part
  first** (`Quatd(real, i, j, k)`), so `[0,0,0,1.0]` is actually parsed as
  `Quatd(real=0, i=0, j=0, k=1.0)`, a **180° rotation about Z**. Confirmed directly:
  `Quatd(0,0,0,1.0)` → rotation matrix `diag(-1,-1,1)`. Confirmed live in a running
  stage too (`GetLocalToWorldTransform` on the terrain prim): the obstacle mesh is
  genuinely rotated 180° about Z relative to what the yaml/comments ("same position as
  before") intended.
  - **Checked whether this breaks the policy's observations vs. physics: it doesn't.**
    Sampled the actual mesh top-surface height and the `.map` heightmap value at the
    same world coordinates (using the *confirmed real* 180°-rotated transform) against
    the robot's actual physics-settled height from a teleop log: all three agree
    (mesh≈0.5/0.765, map≈0.475/0.775, robot root_z≈0.6-0.7/0.85-0.88, matching the same
    ground→box-top step transition). So whatever generated the `.map` file used the
    same (buggy) placement consistently — no observation/physics mismatch. Real bug,
    worth fixing for code clarity/future terrain authoring, but **not the cause of the
    chatter/stall**.
- **Real mesh corruption found, but spatially far from the stall location.**
  Measured triangle quality across all 415,836 triangles in `cur_mixed.usd`: 2.7%
  (11,212) are "needle" triangles (aspect ratio >50, one with aspect ~3 billion — a
  single triangle's edge spans the full 50m terrain extent). Confirmed spatially
  clustered (histogram), not uniform. **But: zero needle triangles exist in the robot's
  actual stuck region** (checked directly — max aspect ratio there is 5.55, completely
  healthy). The corrupted area is ~30-40m away. Real bug, worth fixing, but **not the
  cause of this specific symptom**.
- **New finding: the obstacle the robot stalls on is a simple box with a sharp step,
  and chatter is dramatically worse on top of it than on the ground before it.**
  Dumped the actual mesh triangles in the robot's local operating region: it's a
  literal box — flat top (2 triangles, 4.5 m² each) at world z=0.765, flat ground
  before it (1 giant triangle, 45 m²) at world z=0.5, connected by a vertical wall — a
  clean, simple, ~0.265m step, nothing exotic. Checked the FL1 wheel-velocity-limit
  chatter rate (steps at the ±8.15 rad/s extreme) binned by world x-position against
  this geometry: **0.15-0.32 on top of the box (world x<23) vs. 0.003-0.08 just before
  it (world x>23.5)** — a 50-100x difference, and elevated broadly across the whole box
  top, not just right at the edge. This rules out "coarse mesh triangles cause
  chatter" as the mechanism — the chatter-free ground plane's triangle (45 m²) is
  *larger* than the chatter-prone box-top triangles (4.5 m²), the opposite of what that
  theory predicts. The distinguishing factor instead looks like simply being on an
  isolated, elevated, bounded platform (3m×3m) versus a large continuous ground plane —
  mechanism not yet pinned down further (candidates: wheelbase/flipper span
  interacting with the platform's nearby edges on multiple sides at once; a residual
  contact-solver effect from having just crossed the step that doesn't fully settle).

**11. Confirmed NOT terrain-specific.** Same chatter signature reproduced on a
different `cur_mixed` obstacle (diagonal mound — could approach but not climb, "lack
of thrust") and on `cur_waves` (no movement at all). Both: vx~0.01-0.02 m/s, wheel
vel spikes to ±50-65 rad/s, flipper pivot torque chattering sign every frame
(±200-500 N·m, occasionally pinned at the 1000 N·m limit ~2% of steps — not the main
constraint). Same mechanism, different obstacles → this is general, not a property of
one bad mesh. (Used the new `--spawn_index`/`--list_spawns` teleop flags to target
these specific spots.)

**12. Mechanism audit — leading hypothesis: missing wheel armature (numerically stiff
DOF fighting the contact solver).**
- Contact material: no soft/compliant contact configured anywhere (no
  `PhysxMaterialAPI` compliant stiffness/damping set) — contacts are standard rigid
  PhysX, just friction coefficient (multiply combine) + contact/rest offset + zero
  restitution everywhere. Nothing exotic here.
- GPU contact buffers (2-4x default, sized for thousands of envs): nowhere near
  capacity at num_envs=1 — ruled out as a teleop-specific cause.
- **Wheel DOF is numerically stiff.** Wheel1: inertia ≈0.005 kg·m², velocity-drive
  damping=100 N·m/(rad/s) → natural response time ≈50µs — **~100x faster than one
  sim_dt substep (5ms)**. The flipper *pivot* actuator sets `armature=100`
  specifically to damp this kind of stiffness; the wheel *spin* actuator sets none
  (confirmed: no armature attribute anywhere, on either FTR or MARV, defaults to 0).
  A DOF this stiff, with no armature, fighting a contact constraint every substep
  (drive says "spin at target ω", contact says "zero slip at the patch") is a classic
  setup for exactly the frame-to-frame chatter measured throughout #2/#4/#7/#11. This
  also explains the otherwise-confusing #9 result (more velocity iterations made it
  *worse*) — more iterations converge a stiff/undamped system more aggressively
  toward its oscillation, not away from it; armature is the actual fix for stiffness,
  iteration count isn't.
- Added `--wheel_armature` to `teleop.py`. **Tested at 0.05 — REJECTED.** Box-obstacle
  run, same metrics as baseline: glitch rate 0.0158 vs 0.0150, max|wheel vel| 88 vs 79,
  vx mean 0.037 vs 0.044. No real change, matches reporter's own impression. Armature
  is not the fix.

**13. FIX CONFIRMED AND APPLIED: wheel actuator `stiffness=1` → `0`.**
Wheel actuator had nonzero `stiffness=1` but the code never calls
`set_joint_position_target` for wheels — only `set_joint_velocity_target` — so the
position-target side of the PD drive was stuck at 0 from reset forever while the wheel
spins continuously, creating a phantom restoring torque growing with accumulated
rotation. Tested via `--wheel_stiffness 0.0` on the same box obstacle:

| | stiffness=1 (baseline) | stiffness=0 |
|---|---|---|
| vx mean (fwd cmd), whole session | 0.044 | 0.056 (+27%) |
| glitch rate, whole session | 0.0150 | 0.0126 (-16%) |
| **glitch rate, box top away from edges (x:20.5-23)** | **0.15-0.32** | **0.003-0.027** |
| glitch rate, ground before box (x:23.5-24.5) | 0.003-0.08 | 0.000-0.002 |
| outcome | stuck on box top | **fully crossed the obstacle** |

The whole-session numbers undersell this — they're diluted by averaging over the whole
run. Binned by position, the platform-vs-ground asymmetry that *defined* the original
"stuck on top of the box" symptom (#10/#11) is essentially gone: **a 10-50x reduction**
in the specific zone that mattered, not 16%. The only remaining hot spot (rate 0.234) is
right at the box's *far* edge (~x=20.2, going down off the obstacle) — a discrete
edge-transition event, qualitatively different from the broad sustained on-platform
chatter this fix eliminated. Not a complete fix — vx is still far below the 0.95
commanded, pivot torque still saturates 1-3% of steps — but real
and consistent. **Applied permanently**: `stiffness=0` in both `MARV_CFG`
(`ftr_envs/assets/marv.py`) and `FTR_CFG` (`ftr_envs/assets/ftr.py`, same bug present
there too) `flipper_wheel` actuators.

**14. Why vx is still low even in the now-calm zones: stick-slip, not a clean
oscillation.** Reporter noticed periodic forward-lag-pushback while driving. Checked
directly on the calm box-top zone (stiffness=0): real backward micro-slips exist (2.1%
of steps), and there are "stuck" bouts up to 2.8s interspersed with bursts of faster
rolling (net 1s progress starts ~0.02 m/s, settles to ~0.08-0.13 m/s after ~3s) — but
autocorrelation/FFT on velocity found **no clean dominant period** (checked against
wheel-rotation period 0.77s specifically — not it). This is the signature of
stick-slip (static friction periodically exceeding drive thrust) rather than a fixed
mechanical/numerical beat. Friction itself has never been tested as a lever
(`wheel_material_friction=6.0`, deliberately high from an earlier slope-sliding fix).
Added `--wheel_friction` / `--terrain_friction` to `teleop.py`. **Not yet tested.**

**15. Friction tested — mixed, not a fix.** `wheel_friction=3.0`: backward micro-slip
rate dropped 6.8%→3.1%, but vx mean unchanged (0.044 vs 0.056 baseline) and worst
single stuck bout got *longer* (19.3s vs 7.2s). `terrain_friction=0.7`: same backward
rate drop (3.1%), vx ~flat (0.052), stuck bouts a bit shorter (max 9.1s). Matches
reporter's impression ("little change" / "a bit better but stuck bouts still there").
Friction is a real contributor to the backward-slip frequency but not the main
bottleneck.

**16. Flipper-motion speed-bursts on the box top — confirmed real, but small.**
Reporter noticed speed bursts after flipper movement even on the flat box top.
Checked: lagged correlation between flipper-angle change and `|vx|` peaks at lag
~8-10 steps (~0.2-0.25s) at r≈0.22 — real and consistent with a slight delay (motion
helps, then effect persists briefly), but the magnitude is small: mean `|vx|` 0.089
during flipper movement vs 0.081 static, ~10% relative. Not the dominant driver of the
low overall vx.

**17. DECISIVE: same box obstacle, FTR vs MARV head-to-head — confirmed MARV-specific,
not engine-wide.** Pinned FTR to the identical box (`teleop_ftr.yaml` `spawn_index: 0`,
same terrain, same `stiffness=0` fix, same teleop pipeline). Reporter crossed it twice
in 19 seconds. Measured:

| | MARV (box-top zone) | FTR (box-top zone) |
|---|---|---|
| vx mean (fwd cmd) | 0.05-0.08 | **0.68** (≈70-75% of commanded) |
| glitch rate (per wheel-step) | 0.003-0.03 | **0.0009** (negligible) |

Roughly a 10x difference, same engine/terrain/pipeline. This rules out a generic
PhysX/solver-level explanation definitively — it's specific to MARV's own setup.
Checked the most obvious candidate, wheel mass: FTR's runtime wheel mass is 1.0 kg
(`flipper_wheel_render_mass` override in `ftr_env.py`, not MARV's baked-in 0.73 kg for
wheel1) — only ~37% different, and redoing the stiffness/time-constant math from #12
with 1.0 kg gives ~40µs for FTR vs ~50µs for MARV, the same order of magnitude — not
enough to explain a 10x gap. Total vehicle mass differs more (FTR ~160kg authored
vs MARV ~58kg), but that points the wrong direction for a "MARV too light" story (a
heavier vehicle would intuitively be harder to get moving, not easier). Mass is not
the answer; the real differentiator is still unidentified.

**18. Collision-radius taper theory: briefly reopened, then re-closed correctly.**
Static FTR USD shows uniform 0.09 wheel radius (unlike MARV's 0.1165→0.078 taper),
which looked like a real structural difference. **But this is stale** —
`FtrWheelArticulation.set_robot_env` calls `set_render_radius` at scene setup,
overwriting the collision cylinder radius to `linspace(drive_wheel_radius=0.1165,
auxiliary_wheel_radius=0.078, 5)` — confirmed those exact values are FTR's actual
`render_config` (`ftr_env.py` line ~121). So FTR's *live* collision geometry tapers
identically to MARV's; the static-file value is a pre-override default, not what's
simulated. (Second time static-USD inspection has been misleading this session —
runtime overrides in `set_robot_env` make file-on-disk numbers unreliable in general.)
Added `_print_live_wheel_geometry()` to `teleop.py` (prints actual runtime
radius/contactOffset/restOffset for the front-left flipper's 5 wheels once at
startup) so future FTR/MARV comparisons use ground truth, not static files. **Not yet
captured for either robot — will show up automatically on the next run of either
config.**

Also compared wheel torque/velocity smoothness directly (same box-top zone): FTR's
wheel velocities are smooth and stable (e.g. FL1 mean≈median≈6.08, low variance)
vs MARV's wild swings (mean/median ratio hugely skewed by spikes). FTR's flipper
pivot torque *also* saturates at the 1000 N·m limit on this obstacle (mean|.|=282,
max=1000) — confirms again that pivot saturation alone doesn't block crossing for
either robot, consistent with #7/#13's negative effort-limit result.

**19. New symptom found and confirmed MARV-specific: veering left while driving
straight on the box top.** `cmd_w≈0` but MARV's actual yaw rate is biased
(mean=0.0437, 60% of steps positive vs 40% negative) — 1.08-1.18m of lateral drift
over one box crossing. Checked the obvious causes: flipper pose asymmetry (FL≈FR
within 0.2°, roll≈0 — not it); correlation between flipper angular-velocity
asymmetry and yaw rate, which would implicate the `_flipper_rotation_correction`
term added earlier this session (r=0.02-0.06, negligible — not it). Per-side wheel
velocity diffs exist but flip sign inconsistently across wheel pairs — looks like
noise. **Re-ran the identical box on FTR: essentially zero veering** (y-drift
0.021m, yaw split 48/52, glitch rate 0.0009, crossed in 9.2s at vx=0.675). Same
conditions, same fix, same pipeline. Conclusion: veering is a third symptom of the
same still-unidentified MARV-specific root cause (alongside low vx and chatter
spikes), not a separate bug — random, independent left/right wheel chatter not
perfectly cancelling is the most likely mechanism, pending whatever actually
explains the underlying chatter asymmetry in the first place.

**20. Live contact-offset comparison via the new `_print_live_wheel_geometry` —
ruled out, two more leads closed.** MARV: wheel1 explicit `contactOffset=0.001`
(the earlier fix), wheels 2-5 all `None` (unauthored, PhysX auto-default by size).
FTR: **all 5 wheels show `contactOffset=-inf`/`restOffset=-inf`** — PhysX's explicit
"use auto-computed default" sentinel, structurally equivalent to MARV's `None`.
Critically, FTR's wheel1-equivalent (identical 0.1165 radius) has *never* had any
contact-offset fix and drives fine — directly falsifying "auto-scaled margin too
large for big wheels" as an explanation (if it were, FTR's unfixed same-size wheel
would show it too). Deprioritized: extending the wheel1 contact-offset fix to
wheels 2-5. Also re-confirms #18: since collision-taper geometry is identical
between the two robots and FTR has no issue despite sharing the same wedge, the
wedge itself (already rejected in #6 on different grounds) is doubly ruled out as
the differentiator.

## Open threads / next steps

- **PRIMARY: confirmed MARV-specific (#17), not engine-wide — and now running out of
  parameter-comparison leads.** Ruled out by direct A/B or FTR-vs-MARV comparison:
  torque/effort saturation (#7), wedge-contact/angle/collision-taper geometry (#6,
  #18, #20 — three independent lines all rejecting this), solver velocity iterations
  both directions (#9), terrain mesh corruption and the quaternion bug (#10), wheel
  armature (#13a), wheel/terrain friction (#15), wheel mass (#17), contact
  offset/auto-default margin (#20). Confirmed real but secondary: discrete
  edge-transition chatter (#10/#11), flipper-motion speed bursts (#16), stick-slip
  texture (#14), veering (#19 — a symptom, not a separate cause). The stiffness=0 fix
  (#13) remains the only large, unambiguous improvement found. With geometry, mass,
  friction, and every tested solver/actuator parameter now confirmed equivalent
  between the two robots, the remaining gap is likely in something not yet directly
  measured: actual per-wheel ground contact/normal force during driving (requires
  adding live contact-force sensing — `PhysxContactReportAPI` or similar — to
  `teleop.py`, not yet implemented); or a structural difference in chassis/flipper
  mass *distribution* (not total mass) that changes per-wheel loading even with
  identical wheel hardware.
- Fix the `terrain.py` quaternion bug (`Quatd(*value)` argument order) for code
  clarity/future terrain authoring — confirmed harmless to current behavior, but
  worth fixing since it makes `cur_mixed.yaml`'s prim_config silently not do what it
  says.
- Fix or flag the 11,212 needle triangles (2.7% of `cur_mixed.usd`) — confirmed not the
  cause of this stall, but likely worth cleaning up regardless given the severity
  (aspect ratios up to 3 billion).
- Consider extending the wheel1 contact-offset fix to wheel5 given the bimodal glitch
  data (#2) — lower priority now that #2/#4 are understood to be the same mechanism and
  the contact-offset fix alone didn't resolve it.
- Re-test the wheel-size question (#3) with a method that doesn't trivially remove
  ground contact (e.g. terrain-mesh resolution instead of wheel radius) — lower
  priority; wheel size may still matter for chatter rate even though it couldn't be
  cleanly isolated from "loses contact entirely."

---

## Finding #21 — MARV flipper arm body has overlapping collision geometry with child wheel links

**Discovery (mass distribution / contact force investigation session).**

Reading the MARV URDF xacro (`marv.xacro`, `rendering_target=urdf` branch) and comparing
the generated USD collision geometry against FTR's structure revealed the following:

The MARV flipper arm **link body** (the prim at the revolute pivot joint, with
`flipper_mass = 1e-05 kg` — effectively massless) is imported with its OWN collision
geometry directly from the URDF:

- 3 box shapes spanning the arm length (top belt surface, angled-upper, angled-lower)  
- 1 big cylinder at the pivot end (radius = 0.1165 m = big wheel radius)
- 1 small cylinder at the far end (radius = 0.0780 m = small wheel radius)

The 5 driven **fake-wheel child links** (`front_left_flipper_wheel1` … `wheel5`)
are added as revolute-joint children of the arm body in the URDF — but after URDF
import by Isaac Sim, they become **sibling prims** in the USD stage (flat articulation
hierarchy). Each fake-wheel link has its own cylinder collision prim.

**Problem**: `wheel1` (the biggest, pivot-end fake wheel) has `joint_origin = (0, 0, 0)`
relative to the arm body — i.e., it sits **at exactly the same location** as the arm
body's big cylinder collision shape. Both are radius 0.1165 m, both are cylinders
oriented along Y. `wheel5` similarly overlaps the arm body's small cylinder at the
far end.

This creates **two rigid bodies at the same position both contacting terrain**:
- Arm body: mass ≈ 0 kg, big cylinder at pivot (same position as wheel1)
- Wheel1 link: mass = 0.73 kg, cylinder at pivot

Having two distinct contact constraints at the same geometry vertex, from two bodies
with a ~73000:1 mass ratio, forces PhysX's velocity solver to process two separate
contact manifolds at the same point. This is a documented source of contact
instability in PhysX — the near-zero-mass body prevents consistent impulse
apportionment, producing velocity spikes (the observed wheel1 "glitch") and
solver oscillation (the jerky/sluggish motion).

**Fix**: `MarvWheelArticulation.set_robot_env` now calls `disable_collision_geometry()`
on each flipper arm link prim (`{container}/front_left_flipper`, etc.), disabling ALL
collision shapes on the arm body (walks the prim subtree, sets
`physics:collisionEnabled = False` on each prim carrying `UsdPhysics.CollisionAPI`).
The 5 driven wheel cylinders (on the child-link sibling prims) are **not** under those
prim paths in the USD stage, so they remain active.

Net result: only the 5 fake-wheel cylinder prims contact terrain per flipper. These
already span the full arm length (wheel1 at pivot, wheel5 at far end, wheels 2-4
evenly spaced between) and adequately approximate the continuous belt contact.

**Config**: `disable_flipper_arm_collision: true` (default, set in `FtrEnvCfg` and
propagated to `robot_config`). Expose via `env_cfg_overrides` in config YAML, or via
`--no_disable_flipper_arm_collision` flag in `teleop.sh`.

**Validated** (teleop_wheel1_20260627_000544.csv, 3005 steps, 75 s sim time):

| Metric | Before fix | After fix | Change |
|--------|-----------|-----------|--------|
| Glitch rate (all wheels) | 1.85% | 0.54% | 3.4× reduction |
| Median vx (driving steps) | 0.045 m/s | 0.102 m/s | 2.3× improvement |
| Max vx | 0.570 m/s | 1.011 m/s | 1.8× (exceeds FTR teleop 0.675 m/s) |
| Phantom torque growth | yes (first fix eliminated it) | none (torques decrease 84→35 N·m avg over session) | clean |

User subjective assessment: "felt a thousand times better."

Remaining glitches (0.54%) have SHIFTED from wheel1-dominated to **wheel5-dominated**
(FL5=64, FR5=48, RL5=40 vs FL1=19, FR1=20 before fix). Wheel5 is the small-radius
(0.078 m) wheel at the flipper tip — it experiences lever-arm amplification of the
contact forces from the flipper's far end. Wheel5 also shows the highest torques
(max 9764 N·m for FL5). This is a different mechanism from the arm-body overlap that
was causing wheel1 instability. Possibly addressable by extending the contact-offset
fix to wheel5 (was deprioritized in Finding #2/#4).

---

## Unrelated: a step toward closing the sim-to-real gap

While investigating the "moving the flippers seems to help forward progress" anecdote
(the user noted the real MARV's flippers behave like a continuous belt — angling a
flipper displaces the track relative to the wheels, not necessarily relative to the
ground), found a real missing term in the simulated kinematics:
`FtrWheelArticulation.set_right_and_left_velocities` (and `set_v_w`) computed every
wheel's commanded spin rate purely from the desired chassis `(v, w)`, with no
dependency on the flipper arm's own angle or angular velocity. Since every wheel is
rigidly mounted to its flipper arm (not a free belt), an actively-rotating arm moves
each wheel's center through space — a real ground-contact velocity contribution of
`r · sin(θ) · θ̇` (r = pivot-to-wheel distance, θ = the flipper's raw joint angle) that
was previously uncompensated, forcing artificial slip whenever a flipper is in motion.

Implemented `FtrWheelArticulation._flipper_rotation_correction()` (subtracted from the
commanded surface speed before converting to a wheel spin target in
`set_right_and_left_velocities`), with the corresponding per-wheel pivot-distance
geometry (`load_wheel_pivot_distances`, mirroring each subclass's existing
`load_all_wheel_radius` layout) and per-wheel parent-flipper DOF mapping
(`fl_wheel_flipper_dof`/`fr_wheel_flipper_dof`, built in `find_idx()` for both FTR and
MARV, including the `legacy_ftr_turning` branch). This isn't tied to the MARV bouncing
investigation specifically — it's a general correction that makes the simulated
wheel-on-rigid-arm model behave closer to how the real continuous-belt flipper does
during active motion. **Not yet validated in sim** — the sign convention (assumed to
match the uniform +Y world-frame axis already verified for the wheel spin joints) needs
empirical confirmation; flip the sign in `_flipper_rotation_correction` if it makes
things worse rather than better.
