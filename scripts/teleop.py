"""FTR/MARV Xbox gamepad teleoperation in Isaac Lab — config-driven wrapper.

This is a MARV_RL-side fork of src/FTR-Benchmark/scripts/ftr_algo/teleop.py with two
fixes/additions on top of the original:

  1. Episode timeout fix: the task's default episode_length_s=30 (with decimation=5,
     sim_dt=0.005) truncates and respawns the env every 1200 steps, which cuts off a
     manual driving/climbing session right when it gets interesting. This script
     overrides env_cfg.episode_length_s (configurable, defaults to a very large value
     that effectively disables the timeout) before the env is created.
  2. wheel1 glitch tracking: wheel1 (the lead, big-radius wheel on every flipper)
     intermittently gets its joint velocity kicked to a large negative value under
     load. This script reads each wheel1 joint's velocity every step, flags anomalies,
     prints them live, and logs everything to a CSV for later analysis.

All settings are driven by a YAML config (configs/teleop_marv.yaml, configs/teleop_ftr.yaml),
loaded via --config or the CONFIG env var (see scripts/teleop.sh). Any CLI flag overrides
the corresponding config value.

Launch via the wrapper (recommended — sets PYTHONPATH/conda env):
  CONFIG=teleop_marv.yaml bash scripts/teleop.sh

Launch directly:
  conda run -n isaaclab python scripts/teleop.py --config configs/teleop_marv.yaml
"""

# Isaac Sim AppLauncher MUST be initialised before any omni/carb imports.
import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import yaml

from omni.isaac.lab.app import AppLauncher

_WS_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _WS_ROOT / "configs" / "teleop_marv.yaml"


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    path = Path(path)
    if not path.is_absolute():
        # allow bare filenames resolved against configs/, same convention as CONFIG=foo.yaml elsewhere
        candidate = _WS_ROOT / "configs" / path.name
        path = candidate if candidate.exists() else path
    if not path.exists():
        raise FileNotFoundError(f"Teleop config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--config", type=str, default=os.environ.get("CONFIG"))
_pre_args, _remaining_argv = _pre_parser.parse_known_args()
_cfg_path = _pre_args.config
if _cfg_path is None and _DEFAULT_CONFIG.exists():
    _cfg_path = str(_DEFAULT_CONFIG)
_file_cfg = _load_config(Path(_cfg_path) if _cfg_path else None)

parser = argparse.ArgumentParser(description="FTR/MARV Xbox gamepad teleoperation in Isaac Lab.")
parser.add_argument("--config", type=str, default=_pre_args.config,
                     help="YAML config to load defaults from (overridable by other flags).")
parser.add_argument("--num_envs", type=int, default=_file_cfg.get("num_envs", 1),
                     help="Number of environments to simulate (use 1 for teleop).")
parser.add_argument("--disable_fabric", action="store_true", default=False,
                     help="Disable fabric and use USD I/O (slower, for debugging).")
parser.add_argument("--terrain", type=str, default=_file_cfg.get("terrain", "cur_mixed"),
                     help="Terrain name (default: cur_mixed). Use 'ground' for flat testing.")
parser.add_argument("--spawn_index", type=int, default=_file_cfg.get("spawn_index", None),
                     help="Pin every reset (initial and any mid-session respawn) to this "
                          "birth/spawn-point index instead of the env's normal round-robin "
                          "cycling through terrain_cfg.birth. Use --list_spawns to see the "
                          "available indices and their start/target points first.")
parser.add_argument("--list_spawns", action="store_true", default=False,
                     help="Print all available birth/spawn points for --terrain and exit "
                          "(no sim launched).")
parser.add_argument("--js", type=str, default=_file_cfg.get("js", "/dev/input/js0"),
                     help="Joystick device path.")
parser.add_argument("--robot_type", type=str, default=_file_cfg.get("robot_type", "ftr"),
                     choices=["ftr", "marv"], help="Robot model to simulate.")
parser.add_argument("--v_sensitivity", type=float, default=_file_cfg.get("v_sensitivity", 0.95))
parser.add_argument("--w_sensitivity", type=float, default=_file_cfg.get("w_sensitivity", 1.5))
parser.add_argument("--flipper_sensitivity", type=float, default=_file_cfg.get("flipper_sensitivity", 0.4))
parser.add_argument("--dead_zone", type=float, default=_file_cfg.get("dead_zone", 0.05))
parser.add_argument("--episode_length_s", type=float, default=_file_cfg.get("episode_length_s", 1_000_000.0),
                     help="Overrides the task's episode_length_s. Large value disables the "
                          "default 1200-step timeout/respawn so manual sessions aren't cut off.")
parser.add_argument("--track_wheel1_glitch", action=argparse.BooleanOptionalAction,
                     default=_file_cfg.get("track_wheel1_glitch", True),
                     help="Log/flag wheel1 joint-velocity glitches every step.")
parser.add_argument("--glitch_vel_threshold", type=float,
                     default=_file_cfg.get("glitch_vel_threshold", -10.0),
                     help="rad/s — a wheel1 joint velocity below this is flagged as a glitch.")
parser.add_argument("--glitch_log", type=str, default=_file_cfg.get("glitch_log", None),
                     help="CSV path for wheel1 glitch tracking. Defaults to "
                          "logs/teleop_wheel1_<timestamp>.csv")
parser.add_argument("--wheel1_radius_scale", type=float,
                     default=_file_cfg.get("wheel1_radius_scale", 1.0),
                     help="MARV only: scales wheel1's collision-only radius (kinematic "
                          "radius used for velocity targets is unaffected). 1.0 = unchanged. "
                          "Use e.g. 0.7 to test whether wheel1's larger contact geometry "
                          "(vs FTR's smaller equivalent) makes it snag terrain more often.")
parser.add_argument("--no_disable_flipper_arm_collision", action="store_true",
                     default=False,
                     help="MARV only: disable the fix that removes the flipper arm body's own "
                          "collision geometry. By default the arm body collision (3 boxes + 2 "
                          "end cylinders from the URDF) is disabled because it exactly overlaps "
                          "the driven child wheel-link cylinders at the pivot and far end, "
                          "creating two contact bodies at the same point simultaneously. "
                          "Pass this flag to revert to the original overlapping geometry.")
parser.add_argument("--wheel_armature", type=float,
                     default=_file_cfg.get("wheel_armature", 0.0),
                     help="flipper_wheel actuator armature (currently unset -> 0.0 in both "
                          "MARV_CFG/FTR_CFG; the flipper pivot actuator sets armature=100 but "
                          "the wheel spin actuator does not). Wheel1's own rotational inertia "
                          "is ~0.005 kg*m^2 against a velocity-drive damping gain of 100 "
                          "N*m/(rad/s) -> ~50us natural response time, ~100x faster than one "
                          "sim_dt substep — a numerically stiff DOF that can fight the contact "
                          "solver every substep. Armature adds artificial inertia to damp this. "
                          "Try e.g. 0.01-0.1.")
parser.add_argument("--wheel_stiffness", type=float,
                     default=_file_cfg.get("wheel_stiffness", None),
                     help="flipper_wheel actuator stiffness override (default in MARV_CFG/"
                          "FTR_CFG is 1, nonzero). The code only ever calls "
                          "set_joint_velocity_target for wheels, never set_joint_position_target, "
                          "so the position-target side of the PD drive is stuck at 0 from reset "
                          "forever while the wheel spins continuously — with stiffness=1 this is "
                          "a phantom restoring torque that grows with accumulated rotation "
                          "(~176-240 N*m after ~125s of driving in one log). Try 0.0 to eliminate "
                          "it entirely (pure velocity control).")
parser.add_argument("--wheel_friction", type=float, default=_file_cfg.get("wheel_friction", None),
                     help="Overrides env_cfg.wheel_material_friction (default 6.0 in "
                          "best_long_config_marv.yaml, combine_mode=multiply against terrain "
                          "friction). Untested lever for the stick-slip texture seen on calm "
                          "flat ground (brief 'stuck' bouts up to ~2.8s, rare backward "
                          "micro-slips, bursty rather than smooth forward progress).")
parser.add_argument("--terrain_friction", type=float, default=_file_cfg.get("terrain_friction", None),
                     help="Overrides both env_cfg.terrain_static_friction and "
                          "terrain_dynamic_friction (default 0.7/0.9 area).")
# Fix: parse_env_cfg's registered defaults (FTR_SIM_CFG / MARV_CFG's articulation_props)
# are NOT what train_ftr.py/eval_ftr.py actually run with — every training/eval script
# overrides sim.dt, the per-articulation solver iteration counts, and
# max_depenetration_velocity from the run's config.yaml. Teleop was silently using the
# untuned registered defaults (dt=0.0025, position=16, velocity=4, depenetration=0.15)
# instead of the Optuna-tuned values the policy was actually trained/evaluated with
# (best_long_config_marv.yaml: dt=0.005, position=5, velocity=1, depenetration=0.1235),
# which is a plausible source of the extra bouncing/depenetration seen only in teleop.
parser.add_argument("--solver_position_iterations", type=int,
                     default=_file_cfg.get("solver_position_iterations", 5))
parser.add_argument("--solver_velocity_iterations", type=int,
                     default=_file_cfg.get("solver_velocity_iterations", 1))
parser.add_argument("--sim_dt", type=float, default=_file_cfg.get("sim_dt", 0.005))
parser.add_argument("--max_depenetration_velocity", type=float,
                     default=_file_cfg.get("max_depenetration_velocity", 0.12354395702544713))
AppLauncher.add_app_launcher_args(parser)  # adds --device itself, with its own default
# Apply the config file's device as a default by injecting it ahead of any CLI override
# (AppLauncher owns --device, so we can't redeclare it with our own argparse default).
if "device" in _file_cfg and not any(a == "--device" or a.startswith("--device=") for a in _remaining_argv):
    _remaining_argv = ["--device", str(_file_cfg["device"])] + _remaining_argv
args_cli = parser.parse_args(_remaining_argv)

if args_cli.list_spawns:
    # Terrain/birth-point inspection needs no omni/Isaac Sim imports — skip launching
    # the sim entirely for a fast, no-GPU answer.
    _ftr_root_early = str(_WS_ROOT / "src" / "FTR-Benchmark")
    if _ftr_root_early not in sys.path:
        sys.path.insert(0, _ftr_root_early)
    from ftr_envs.assets.terrain.terrain import Terrain

    terrain = Terrain(args_cli.terrain)
    print(f"{len(terrain.birth)} spawn point(s) for terrain '{args_cli.terrain}':")
    for i, b in enumerate(terrain.birth):
        print(f"  [{i}] start={b['start_point']}  start_orient={b['start_orient']}  "
              f"target={b['target_point']}")
    print("\nUse --spawn_index N (or spawn_index: N in the config) to pin every reset to one.")
    sys.exit(0)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------- post-launch imports ----------
import struct
import threading

import gymnasium as gym
import numpy as np
import omni.isaac.lab_tasks  # noqa: F401 — registers built-in Isaac Lab tasks
import torch
from omni.isaac.lab_tasks.utils import parse_env_cfg

# ftr_envs is not an installed package; add the submodule root to sys.path.
_FTR_ROOT = str(_WS_ROOT / "src" / "FTR-Benchmark")
if _FTR_ROOT not in sys.path:
    sys.path.insert(0, _FTR_ROOT)

import ftr_envs.tasks            # noqa: F401 — registers Ftr-Crossing-Direct-v0
import ftr_envs.utils.omega_conf  # noqa: F401 — OmegaConf resolvers


# ---------- joystick reader ----------

class _LinuxJoystick:
    """Reads Linux joystick events from /dev/input/jsX in a background thread."""

    _FMT = "IhBB"
    _SIZE = struct.calcsize(_FMT)
    _BUTTON = 1
    _AXIS = 2

    def __init__(self, device: str = "/dev/input/js0"):
        try:
            self._fd = open(device, "rb")
        except PermissionError:
            raise RuntimeError(f"Permission denied opening {device}. Run: sudo chmod a+r {device}")
        except FileNotFoundError:
            raise RuntimeError(f"{device} not found. Is the controller connected?")

        self._axes: dict[int, float] = {}
        self._buttons: dict[int, bool] = {}
        self._lock = threading.Lock()
        self._running = True

        self._thread = threading.Thread(target=self._loop, daemon=True, name="js_reader")
        self._thread.start()
        print(f"[FtrGamepad] Reading joystick from {device}", flush=True)

    def _loop(self) -> None:
        while self._running:
            data = self._fd.read(self._SIZE)
            if not data or len(data) < self._SIZE:
                break
            _time, value, ev_type, number = struct.unpack(self._FMT, data)
            ev_type &= ~0x80  # strip JS_EVENT_INIT
            with self._lock:
                if ev_type == self._AXIS:
                    self._axes[number] = value / 32767.0
                elif ev_type == self._BUTTON:
                    self._buttons[number] = bool(value)

    def axis(self, n: int, default: float = 0.0) -> float:
        with self._lock:
            return self._axes.get(n, default)

    def button(self, n: int, default: bool = False) -> bool:
        with self._lock:
            return self._buttons.get(n, default)

    def close(self) -> None:
        self._running = False
        try:
            self._fd.close()
        except Exception:
            pass


# ---------- gamepad handler ----------

class FtrGamepad:
    """Xbox controller wrapper for the FTR/MARV 6-DOF action space."""

    def __init__(
        self,
        v_sensitivity: float = 0.95,
        w_sensitivity: float = 1.0,
        flipper_sensitivity: float = 0.8,
        dead_zone: float = 0.05,
        device: str = "/dev/input/js0",
    ):
        self.v_sensitivity = v_sensitivity
        self.w_sensitivity = w_sensitivity
        self.flipper_sensitivity = flipper_sensitivity
        self.dead_zone = dead_zone
        self._js = _LinuxJoystick(device)

    def _apply_dz(self, val: float) -> float:
        if abs(val) < self.dead_zone:
            return 0.0
        sign = 1.0 if val > 0.0 else -1.0
        return sign * (abs(val) - self.dead_zone) / (1.0 - self.dead_zone)

    def advance(self) -> np.ndarray:
        """Return the 6-element FTR/MARV action [v, w, fl, fr, rl, rr]."""
        v = self._apply_dz(-self._js.axis(1)) * self.v_sensitivity
        w = self._apply_dz(self._js.axis(0)) * self.w_sensitivity
        fv = self._apply_dz(-self._js.axis(4)) * self.flipper_sensitivity

        lb = self._js.button(4)
        rb = self._js.button(5)
        lt = self._js.axis(2) > 0.0
        rt = self._js.axis(5) > 0.0

        return np.array([
            v, w,
            fv if lb else 0.0,
            fv if rb else 0.0,
            fv if lt else 0.0,
            fv if rt else 0.0,
        ], dtype=np.float32)

    def close(self) -> None:
        self._js.close()

    def __str__(self) -> str:
        return f"FtrGamepad  device='{self._js._fd.name}'"


# ---------- wheel glitch tracking ----------

def _resolve_all_wheel_joints(robot, robot_type: str) -> list[tuple[str, int]]:
    """Return [(wheel_label, joint_idx), ...] for *every* wheel (1..N) on every flipper,
    e.g. FL1..FL5, FR1..FR5, RL1..RL5, RR1..RR5 — not just the lead wheel1 — so we can see
    whether the glitch is specific to wheel1 or shows up on the trailing wheels too.
    """
    if robot_type == "marv":
        pattern = re.compile(r"^(front_left|front_right|rear_left|rear_right)_flipper_wheel(\d+)_j$")
        side_label = {"front_left": "FL", "front_right": "FR", "rear_left": "RL", "rear_right": "RR"}
    else:
        # FTR joint-name prefixes don't match left/right side despite appearances — see the
        # mapping comment in ftr_envs/assets/articulation/ftr.py: LF=front-left, LR=front-right,
        # RL=rear-left, RR=rear-right.
        pattern = re.compile(r"^(LF|LR|RL|RR)(\d+)RevoluteJoint$")
        side_label = {"LF": "FL", "LR": "FR", "RL": "RL", "RR": "RR"}

    out = []
    for name in robot.flipper_wheel_joint_names:
        m = pattern.match(name)
        if m:
            side, num = m.group(1), m.group(2)
            label = f"{side_label[side]}{num}"
            idx = robot.find_joints(name)[0][0]
            out.append((label, idx))
    out.sort(key=lambda t: (t[0][:2], int(t[0][2:])))
    return out


class Wheel1GlitchTracker:
    """Logs every wheel's joint velocity and robot/flipper state every step, and flags
    anomalous negative spikes on any wheel (not just wheel1).
    """

    def __init__(self, env, robot_type: str, threshold: float, log_path: Path):
        self.env = env
        self.robot = env.unwrapped._robot
        self.joints = _resolve_all_wheel_joints(self.robot, robot_type)
        self.threshold = threshold
        self.log_path = log_path
        self.step = 0
        self.glitch_count = 0

        # flipper pivot joints, in flipper_dof_idx_list order: [front_left, front_right,
        # rear_left, rear_right] — see FtrWheelArticulation.find_idx.
        self.pivot_idx = list(self.robot.flipper_dof_idx_list)
        self.pivot_labels = ["FL", "FR", "RL", "RR"]

        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(log_path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(
            ["step", "sim_time", "cmd_v", "cmd_w",
             "actual_lin_vel_x", "actual_lin_vel_y", "actual_ang_vel_z",
             "root_pos_x", "root_pos_y", "root_pos_z",
             "roll_deg", "pitch_deg"]
            + [f"{label}_vel" for label, _ in self.joints]
            + [f"{label}_glitch" for label, _ in self.joints]
            + [f"{label}_torque" for label, _ in self.joints]
            + ["FL_flipper_deg", "FR_flipper_deg", "RL_flipper_deg", "RR_flipper_deg"]
            + [f"{label}_pivot_torque_applied" for label in self.pivot_labels]
            + [f"{label}_pivot_torque_computed" for label in self.pivot_labels]
        )
        print(f"[WheelGlitch] Tracking joints {[l for l, _ in self.joints]} -> {log_path}", flush=True)

    def update(self, cmd_v: float, cmd_w: float, sim_time: float) -> list[str]:
        unwrapped = self.env.unwrapped
        joint_vel = self.robot.data.joint_vel[0]
        vels = [joint_vel[idx].item() for _, idx in self.joints]
        flags = [v < self.threshold for v in vels]

        applied_torque = self.robot.data.applied_torque[0]
        computed_torque = self.robot.data.computed_torque[0]
        wheel_torques = [applied_torque[idx].item() for _, idx in self.joints]
        pivot_applied = [applied_torque[idx].item() for idx in self.pivot_idx]
        pivot_computed = [computed_torque[idx].item() for idx in self.pivot_idx]

        lin_vel = unwrapped.robot_lin_velocities[0]
        ang_vel = self.robot.data.root_ang_vel_b[0]
        root_pos = self.robot.data.root_pos_w[0]
        # roll/pitch from the projected-gravity vector
        g = self.robot.projected_gravity[0]
        roll = float(np.degrees(np.arctan2(g[1].item(), -g[2].item())))
        pitch = float(np.degrees(np.arctan2(-g[0].item(), -g[2].item())))
        flippers_deg = [float(np.degrees(f)) for f in unwrapped.flipper_positions[0].tolist()]

        row = (
            [self.step, sim_time, cmd_v, cmd_w,
             lin_vel[0].item(), lin_vel[1].item(), ang_vel[2].item(),
             root_pos[0].item(), root_pos[1].item(), root_pos[2].item(),
             roll, pitch]
            + vels + flags + wheel_torques + flippers_deg + pivot_applied + pivot_computed
        )
        self._writer.writerow(row)
        self.step += 1

        flagged = [label for (label, _), flag in zip(self.joints, flags) if flag]
        if flagged:
            self.glitch_count += 1
        return flagged

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()
        print(
            f"[WheelGlitch] {self.glitch_count} glitch step(s) out of {self.step} "
            f"logged to {self.log_path}",
            flush=True,
        )


# ---------- helpers ----------

def print_controls() -> None:
    print()
    print("=" * 56)
    print("  FTR/MARV Teleop — Xbox Gamepad (config-driven)")
    print("=" * 56)
    print("  DRIVING")
    print("    Left  stick Y  : linear  velocity  (fwd/bwd)")
    print("    Left  stick X  : angular velocity  (turn L/R)")
    print()
    print("  FLIPPERS  (right stick Y = speed; hold to select)")
    print("    LB  (hold)     : front-left  flipper")
    print("    RB  (hold)     : front-right flipper")
    print("    LT  (> 50%)    : rear-left   flipper")
    print("    RT  (> 50%)    : rear-right  flipper")
    print("    Right stick UP : positive delta (extends up)")
    print()
    print("  Episode timeout disabled for this session (episode_length_s override).")
    print("  Status printed every ~1 s  |  Close viewport to quit")
    print("=" * 56)
    print()


def _print_live_wheel_geometry(env, robot_type: str) -> None:
    """One-time diagnostic: dump each front-left-flipper wheel's *actual runtime*
    collision radius and contact/rest offset, read directly from the live USD stage
    after set_robot_env()/load_all_wheel_radius() have run. Static-USD inspection is
    unreliable here — set_render_radius/set_collision_offsets overwrite these values
    at scene setup, so the file-on-disk numbers are stale defaults, not what's actually
    simulated. Logged once at startup so FTR vs MARV runs can be compared on real numbers.
    """
    import omni.usd
    from pxr import PhysxSchema, UsdGeom

    robot = env.unwrapped._robot
    container = robot.robot_prim_path
    if robot_type == "marv":
        paths = [f"{container}/front_left_flipper_wheel{i}/collisions" for i in range(1, 6)]
    else:
        paths = [f"{container}/flipper_list/front_left_wheel/FL{i}/FlipperRender" for i in range(1, 6)]

    stage = omni.usd.get_context().get_stage()
    print(f"[WheelGeometry] live front-left-flipper wheel collision geometry ({robot_type}):", flush=True)
    for i, path in enumerate(paths, start=1):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"  wheel{i}: {path} -> invalid prim", flush=True)
            continue
        radius = UsdGeom.Cylinder(prim).GetRadiusAttr().Get()
        api = PhysxSchema.PhysxCollisionAPI(prim)
        contact_offset = api.GetContactOffsetAttr().Get() if api else None
        rest_offset = api.GetRestOffsetAttr().Get() if api else None
        print(f"  wheel{i}: radius={radius} contactOffset={contact_offset} restOffset={rest_offset}",
              flush=True)


# ---------- main ----------

TASK = "Ftr-Crossing-Direct-v0"
STATUS_STEPS = 10


def main() -> None:
    env_cfg = parse_env_cfg(
        TASK,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.terrain_name = args_cli.terrain
    env_cfg.robot_type = args_cli.robot_type
    env_cfg.initial_flipper_range = (0, 0)
    env_cfg.forward_vel_range = (0.0, 0.0)
    # Fix: the task default (episode_length_s=30, decimation=5, sim_dt=0.005) truncates
    # and respawns the env every 1200 steps, cutting off manual teleop sessions.
    env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.wheel1_collision_radius_scale = args_cli.wheel1_radius_scale
    env_cfg.disable_flipper_arm_collision = not args_cli.no_disable_flipper_arm_collision
    if args_cli.wheel_armature != 0.0:
        env_cfg.robot.actuators["flipper_wheel"].armature = args_cli.wheel_armature
    if args_cli.wheel_stiffness is not None:
        env_cfg.robot.actuators["flipper_wheel"].stiffness = args_cli.wheel_stiffness
    if args_cli.wheel_friction is not None:
        env_cfg.wheel_material_friction = args_cli.wheel_friction
    if args_cli.terrain_friction is not None:
        env_cfg.terrain_static_friction = args_cli.terrain_friction
        env_cfg.terrain_dynamic_friction = args_cli.terrain_friction

    # Fix: match the solver/timestep settings train_ftr.py/eval_ftr.py actually run with
    # (see argparse help above) instead of parse_env_cfg's untuned registered defaults.
    env_cfg.sim.dt = args_cli.sim_dt
    env_cfg.robot.spawn.rigid_props.max_depenetration_velocity = args_cli.max_depenetration_velocity
    env_cfg.robot.spawn.articulation_props.solver_position_iteration_count = args_cli.solver_position_iterations
    env_cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = args_cli.solver_velocity_iterations
    env_cfg.sim.physx.min_position_iteration_count = args_cli.solver_position_iterations
    env_cfg.sim.physx.max_velocity_iteration_count = args_cli.solver_velocity_iterations

    env = gym.make(TASK, cfg=env_cfg)

    if args_cli.spawn_index is not None:
        # Bypass the env's normal round-robin cycling through terrain_cfg.birth
        # (ftr_env.py's _prepare_reset_info/_reset_info_generate) and pin every reset —
        # initial and any mid-session respawn alike — to one fixed spawn point, so the
        # same obstacle/terrain section can be tested repeatably across runs.
        unwrapped = env.unwrapped
        spawns = unwrapped._reset_info
        idx = args_cli.spawn_index % len(spawns)
        if args_cli.spawn_index != idx:
            print(f"[INFO] spawn_index {args_cli.spawn_index} wrapped to {idx} "
                  f"({len(spawns)} spawn points available; use --list_spawns to see them).",
                  flush=True)
        unwrapped._reset_info_generate = lambda: spawns[idx]
        print(f"[INFO] Pinned every reset to spawn_index {idx}: "
              f"start={spawns[idx]['start_point'].tolist()} "
              f"target={spawns[idx]['target_point'].tolist()}", flush=True)

    try:
        gamepad = FtrGamepad(
            v_sensitivity=args_cli.v_sensitivity,
            w_sensitivity=args_cli.w_sensitivity,
            flipper_sensitivity=args_cli.flipper_sensitivity,
            dead_zone=args_cli.dead_zone,
            device=args_cli.js,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", flush=True)
        env.close()
        return

    print(gamepad)
    print_controls()

    # find_idx() (which populates flipper_dof_idx_list/fl_indices/fr_indices) only runs on
    # reset(), so the env must be reset before the tracker can resolve joint indices.
    obs, _ = env.reset()
    step = 0

    _print_live_wheel_geometry(env, args_cli.robot_type)

    glitch_tracker = None
    if args_cli.track_wheel1_glitch:
        glitch_log = args_cli.glitch_log
        if glitch_log is None:
            glitch_log = _WS_ROOT / "logs" / f"teleop_wheel1_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        glitch_tracker = Wheel1GlitchTracker(
            env, args_cli.robot_type, args_cli.glitch_vel_threshold, Path(glitch_log)
        )

    while simulation_app.is_running():
        cmd = gamepad.advance()
        actions = torch.tensor(cmd, dtype=torch.float32, device=args_cli.device).unsqueeze(0)
        obs, _rew, terminated, truncated, _info = env.step(actions)

        if glitch_tracker is not None:
            sim_time = step * env.unwrapped.physics_dt * env.unwrapped.cfg.decimation
            flagged = glitch_tracker.update(float(cmd[0]), float(cmd[1]), sim_time)
            if flagged:
                print(f"[{step:6d}] [WheelGlitch] {flagged}", file=sys.stderr, flush=True)

        if terminated.any() or truncated.any():
            print("[INFO] Episode ended — resetting.", flush=True)
            obs, _ = env.reset()

        step += 1
        if step % STATUS_STEPS == 0:
            vel = env.unwrapped.robot_lin_velocities[0, 0].item()
            flips_deg = [
                round(np.rad2deg(f), 1)
                for f in env.unwrapped.flipper_positions[0].tolist()
            ]
            js = gamepad._js
            rew_info = env.unwrapped.extras.get("reward_components", {})
            accel_mag = rew_info.get("shock/accel_magnitude", float("nan"))
            shock_norm = rew_info.get("shock/shock_normalised", float("nan"))
            print(
                f"[{step:6d}]  cmd=[v={cmd[0]:+.2f} w={cmd[1]:+.2f}]  "
                f"v_actual={vel:+.2f} m/s  flippers={flips_deg} deg  "
                f"shock={accel_mag:.1f} m/s²  shock_norm={shock_norm:.3f}  "
                f"ax0-5=[{js.axis(0):+.2f} {js.axis(1):+.2f} {js.axis(2):+.2f} "
                f"{js.axis(3):+.2f} {js.axis(4):+.2f} {js.axis(5):+.2f}]  "
                f"btn=[{js.button(4)} {js.button(5)}]",
                file=sys.stderr, flush=True,
            )

    if glitch_tracker is not None:
        glitch_tracker.close()
    gamepad.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
