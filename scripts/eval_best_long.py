"""Native physics evaluation of an FTR-trained policy.

Loads a policy trained with train_ftr.py (Isaac Sim) and evaluates it in the
native PyTorch physics engine.  Reports success and failure rates and opens a
SimView browser visualisation when --vis is passed.

Observation bridge (968-D → 966-D):
  FtrCompatObservation produces 968-D obs; this script slices off the last 2
  dims (prev_action[6:8]) so the loaded 966-D policy encoder receives the right
  input shape.  Those 2 dims are the least-informative part of the observation
  (extra track-velocity history), so the approximation is minor.

Action bridge (6-D → 8-D):
  The FTR policy outputs [v, w, fl, fr, rl, rr] in [-1, 1].
  The native env expects [4 track vels (m/s), 4 flipper vels (rad/s)].
  Conversion uses the robot model's vw_to_vels() for differential drive.

Usage:
  /home/robot/conda/envs/isaaclab/bin/python scripts/eval_best_long.py \\
      --run_dir experiments/best_long/attempt_0 \\
      --num_robots 64 --max_steps 600 --terrain cur_mixed --vis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

import flipper_training  # noqa: F401 — registers OmegaConf resolvers
from flipper_training.experiments.ppo.ftr_policy_inference_module import FtrPolicyInferenceModule
from flipper_training.experiments.ppo.train_ftr_compat import _build_env
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.environment.env import Env
from flipper_training.utils.torch_utils import seed_all


_OBS_KEY_COMPAT = "FtrCompatObservation"
_OBS_KEY_POLICY = "FtrFlatObservation"
_OBS_DIM_POLICY = 966


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an FTR-trained policy in native physics.")
    p.add_argument("--run_dir", type=Path, default=Path("experiments/best_long/attempt_0"))
    p.add_argument("--weight", type=str, default="final", help="Checkpoint suffix: 'final' or a step number.")
    p.add_argument("--num_robots", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=None,
                   help="Hard cap on total simulation steps. Defaults to 600 when --eval_episodes "
                        "is not set, or eval_episodes × 800 when it is (giving ~20 s per episode).")
    p.add_argument("--eval_episodes", type=int, default=None,
                   help="Stop once every robot has completed this many episodes. "
                        "--max_steps acts as a safety cap so stuck robots don't block forever.")
    p.add_argument("--terrain", type=str, default="native_mixed",
                   help="Terrain type. Native options: native_mixed, native_flat, native_stairs, native_gaussian, native_trunks. "
                        "FTR options: cur_mixed, cur_base, etc. (default: native_mixed)")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto).")
    p.add_argument("--vis", action="store_true", help="Open SimView browser visualisation after eval.")
    p.add_argument("--vis_robots", type=int, default=None, help="Number of robots to record for vis (default: all).")
    return p.parse_args()


def _load_policy(run_dir: Path, weight: str, device: str) -> FtrPolicyInferenceModule:
    suffix = f"step_{weight}" if weight.isdigit() else weight
    policy_path = run_dir / "weights" / f"policy_{suffix}.pth"
    vecnorm_path = run_dir / "weights" / f"vecnorm_{suffix}.pth"
    return FtrPolicyInferenceModule(
        config_path=run_dir / "config.yaml",
        policy_weights_path=policy_path,
        vecnorm_weights_path=vecnorm_path if vecnorm_path.exists() else None,
        device=device,
        num_actions=6,
    )


def _infer_batch(
    module: FtrPolicyInferenceModule,
    obs_compat: torch.Tensor,  # (B, 968)
) -> torch.Tensor:
    """Run batched policy inference.  Returns (B, 6) action tensor."""
    obs_966 = obs_compat[:, :_OBS_DIM_POLICY].clone()  # (B, 966)
    # Front flipper positions (indices 953:955 = FL, FR) are encoded as
    # (native_angle + fp_max) / (2*fp_max) in FtrCompatObs, where native positive = DOWN.
    # FTR policy expects (1 - native_norm) so that 1.0 = fully up, 0.0 = fully down.
    obs_966[:, 953:955] = 1.0 - obs_966[:, 953:955]
    td = TensorDict(
        {_OBS_KEY_POLICY: obs_966},
        batch_size=[obs_966.shape[0]],
        device=module.device,
    )
    # VecNorm normalises in-place but also returns the TD; capture to be safe.
    td = module.vecnorm(td)
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        td = module.actor(td)
    return td["action"]  # (B, 6)


def _action_ftr_to_native(
    action_6: torch.Tensor,  # (B, 6)  [v, w, fl, fr, rl, rr] in [-1, 1]
    robot_cfg,
) -> torch.Tensor:
    """Convert 6-D FTR action → 8-D native physics action [4 track vels, 4 flipper vels]."""
    v = action_6[:, 0:1] * robot_cfg.v_max   # (B, 1)  linear vel in m/s
    w = action_6[:, 1:2] * robot_cfg.v_max   # (B, 1)  angular vel (same scale, geometry handles it)
    track_vels = robot_cfg.vw_to_vels(v, w).to(action_6.device)   # (B, n_tracks)

    # Flipper vels: policy output [-1,1] scaled to joint velocity limit.
    # Sign convention: FTR (flipper_style=False) encodes positive front flipper as UP,
    # but native physics positive = DOWN for front flippers.  Negate FL, FR; keep RL, RR.
    flipper_max = float(robot_cfg.joint_limits.abs().max())
    flipper_signs = action_6.new_tensor([-1., -1., 1., 1.])  # [FL, FR, RL, RR]
    flipper_vels = action_6[:, 2:6] * flipper_max * flipper_signs  # (B, 4)
    return torch.cat([track_vels, flipper_vels], dim=1)  # (B, 8)


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg.num_robots = args.num_robots
    if args.device:
        cfg.device = args.device
    cfg.terrain = args.terrain
    # Native terrains use a smaller, denser world than the FTR 8 m arena.
    if args.terrain.startswith("native_"):
        OmegaConf.update(cfg, "max_coord", 4.0, merge=False)
        OmegaConf.update(cfg, "grid_res", 0.05, merge=False)
    # Match FTR training control frequency: sim_dt=0.005 × 5 iters = 40 Hz.
    OmegaConf.update(cfg, "engine_opts", {"dt": 0.005, "damping_alpha": 5.0}, merge=False)
    OmegaConf.update(cfg, "engine_iters_per_env_step", 5, merge=False)
    # Disable engine compilation for eval (avoids long first-step warmup).
    OmegaConf.update(cfg, "engine_compile_opts", {}, merge=False)
    # Override objective with eval-friendly settings:
    #   - robot always spawned facing the goal
    #   - goal always reachable without excessive height gain
    #   - goal distance matched to terrain size
    max_coord = float(OmegaConf.select(cfg, "max_coord", default=4.0))
    OmegaConf.update(cfg, "objective_opts", {
        "start_position_orientation": "towards_goal",
        "higher_allowed": 0.5,        # goal at most 0.5 m above start
        "min_dist_to_goal": 2.0,
        "max_dist_to_goal": max_coord * 0.9,
        "max_feasible_roll": 1.2,     # generous termination — let policy try harder
        "max_feasible_pitch": 1.2,
    }, merge=False)

    device_str = str(cfg.device)
    print(f"Loaded config from {run_dir}")
    print(f"  terrain={OmegaConf.select(cfg, 'terrain', default='cur_mixed')}  "
          f"num_robots={args.num_robots}  max_steps={args.max_steps}  device={device_str}")

    # ── Policy ────────────────────────────────────────────────────────────────
    policy_module = _load_policy(run_dir, args.weight, device_str)
    policy_module.actor.eval()
    policy_module.vecnorm.eval()

    # ── Native physics env ───────────────────────────────────────────────────
    rng = seed_all(int(OmegaConf.select(cfg, "seed", default=42)))
    env, device = _build_env(cfg, rng)
    robot_cfg = env.robot_cfg
    if args.vis:
        # Enable derivative output so SimView can show contacts/velocities.
        env.return_derivative = True
        env.observation_spec = env._make_observation_spec()

    if args.max_steps is not None:
        max_steps = args.max_steps
    elif args.eval_episodes is not None:
        max_steps = args.eval_episodes * 800  # ~20 s per episode at 40 Hz
    else:
        max_steps = 600

    if args.eval_episodes is not None:
        stop_desc = f"{args.eval_episodes} episodes/robot (safety cap: {max_steps} steps)"
    else:
        stop_desc = f"{max_steps} steps total"
    print(f"\nRunning eval: {args.num_robots} robots × {stop_desc} "
          f"on {args.terrain} (native physics)")

    # ── SimView setup (optional) ──────────────────────────────────────────────
    simview = None
    vis_n = args.vis_robots if args.vis_robots is not None else args.num_robots
    if args.vis:
        from simview import SimView
        from flipper_training.vis.simview import (
            physics_state_to_simview_body_states,
            simview_bodies_from_robot_config,
            simview_terrain_from_config,
        )
        simview = SimView(
            run_name=f"eval_{run_dir.name}",
            batch_size=vis_n,
            scalar_names=["succeeded", "failed"],
            dt=env.effective_dt,
            collapse=False,
            use_cache=False,
        )
        simview.model.add_terrain(simview_terrain_from_config(env.terrain_cfg))
        for body in simview_bodies_from_robot_config(robot_cfg):
            simview.model.add_body(body)
        for static_obj in env.objective.start_goal_to_simview(env.start, env.goal):
            simview.model.add_static_object(static_obj)

    # ── Manual rollout loop ───────────────────────────────────────────────────
    # The env auto-resets done robots, so one robot can complete multiple episodes.
    # --eval_episodes: run until every robot finishes that many episodes (fair comparison).
    # --max_steps: hard cap (always active as a safety limit).
    n_succeeded = 0
    n_failed = 0
    n_completed = 0
    episodes_per_robot = torch.zeros(args.num_robots, dtype=torch.long, device=device)

    td = env.reset()

    for step in range(max_steps):
        # ── Build and apply action ────────────────────────────────────────────
        obs_compat = td[_OBS_KEY_COMPAT]  # (B, 968)
        action_6 = _infer_batch(policy_module, obs_compat)  # (B, 6)
        native_action = _action_ftr_to_native(action_6, robot_cfg)  # (B, 8)

        td_in = td.copy()
        td_in["action"] = native_action
        td_out = env.step(td_in)

        next_td = td_out["next"]

        # ── Count outcomes ────────────────────────────────────────────────────
        succeeded = next_td["succeeded"].squeeze(-1)  # (B,) bool
        failed = next_td["failed"].squeeze(-1)         # (B,) bool

        done = succeeded | failed
        n_succeeded += int(succeeded.sum())
        n_failed += int(failed.sum())
        n_completed += int(done.sum())
        episodes_per_robot += done.long()

        if args.eval_episodes is not None and episodes_per_robot.min() >= args.eval_episodes:
            td = next_td  # still advance before breaking so SimView gets the final frame
            break

        # ── Record SimView frame ──────────────────────────────────────────────
        if simview is not None:
            curr_state = PhysicsState.from_tensordict(next_td[Env.STATE_KEY])
            curr_state_der = PhysicsStateDer.from_tensordict(next_td[Env.PREV_STATE_DER_KEY])
            curr_state_vis = curr_state[:vis_n]
            curr_state_der_vis = curr_state_der[:vis_n]
            body_states = physics_state_to_simview_body_states(
                robot_cfg,
                curr_state_vis,
                curr_state_der_vis,
                native_action[:vis_n],
            )
            simview.add_state(
                env.effective_dt * step,
                body_states=body_states,
                scalar_values={
                    "succeeded": succeeded[:vis_n].int().tolist(),
                    "failed": failed[:vis_n].int().tolist(),
                },
            )

        # Advance td (env auto-resets done robots).
        td = next_td

    min_ep = int(episodes_per_robot.min())
    max_ep = int(episodes_per_robot.max())
    ep_range = f"{min_ep}" if min_ep == max_ep else f"{min_ep}–{max_ep}"

    if args.eval_episodes is not None:
        total = args.num_robots * args.eval_episodes
        n_timeout = total - n_succeeded - n_failed
        cap_hit = int(episodes_per_robot.min()) < args.eval_episodes
        if cap_hit:
            print(f"  WARNING: safety cap hit — {int((episodes_per_robot < args.eval_episodes).sum())} robots "
                  f"completed fewer than {args.eval_episodes} episodes; timeouts counted as failures.")
        print(f"\nCompleted episodes : {n_completed}  ({ep_range} per robot, {args.num_robots} robots)")
        print(f"Success rate       : {n_succeeded / total:.1%}  ({n_succeeded}/{total})")
        print(f"Failure rate       : {n_failed / total:.1%}  ({n_failed}/{total})")
        print(f"Timeout rate       : {n_timeout / total:.1%}  ({n_timeout}/{total})")
    else:
        total = max(n_completed, 1)
        print(f"\nCompleted episodes : {n_completed}  ({ep_range} per robot, {args.num_robots} robots)")
        print(f"Success rate       : {n_succeeded / total:.1%}  ({n_succeeded}/{total})")
        print(f"Failure rate       : {n_failed / total:.1%}  ({n_failed}/{total})")
        print("(Timeouts excluded — use --eval_episodes for a stricter metric)")

    if simview is not None:
        print("\nOpening SimView — visit the URL shown below in your browser.")
        simview.visualize()


if __name__ == "__main__":
    main()
