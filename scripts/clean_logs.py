#!/usr/bin/env python3
"""Prune old checkpoints, merge SLURM-array-split optuna studies, and delete failed/incomplete runs under logs/.

Before any run's .out/.err are deleted, a classification record is appended to
<logs-dir>/clean_logs_audit.jsonl (same schema as notebooks/slurm_trial_audit.ipynb's
classify_trial()), so that notebook stays usable after cleanup. Never touches active jobs
(--min-age-hours) or your most recent runs (--keep-last). Dry-run makes no changes at all,
including to the audit log.

Checkpoint pruning is trainer-agnostic: it groups every "<family>_step_<n>.pth" by step, so a
kept/deleted slot takes all the files that step wrote with it, whichever rl_module produced
them (policy/vecnorm for marv_rl/hfc/mitriakov/atd3qn, + icm for icmd3qn, + qvalue/cvae for
ctrac, and creps_state alone for creps). Resume-only state -- training_state.pth,
D3QN/ICM-D3QN's weights/replay_buffer.pt, and C-TRAC's job-level replay_buffer/ directory
(~21 GB) -- is removed once the job that could have resumed from it is over.

Optionally (--compact-raw-accel), also shrinks raw_accel.npz/raw_accel_eval.npz debug arrays
from full per-sample dumps (up to ~200MB each) down to a fine-grained histogram + exact
summary stats (a few KB), preserving what scripts/plot_shock_distribution.py actually needs.

Usage:
    python scripts/clean_logs.py [--logs-dir PATH] [--dry-run] [--keep-last N]
                                  [--min-age-hours H] [--eval-min-repeats R]
                                  [--audit-log PATH] [--compact-raw-accel]
                                  [--raw-accel-bins N] [--exclude NAME ...]
                                  [--backup-dir PATH] [-v]

Examples:
    python scripts/clean_logs.py --logs-dir logs --dry-run -v   # preview, no changes
    python scripts/clean_logs.py --logs-dir logs                # apply for real
    python scripts/clean_logs.py --logs-dir logs --compact-raw-accel
    python scripts/clean_logs.py --logs-dir /path/to/logs --dry-run
    python scripts/clean_logs.py --logs-dir logs --exclude train_ctrac_11311704   # leave untouched
    python scripts/clean_logs.py --logs-dir logs --backup-dir /tmp/clean_logs_backup  # copy before delete/prune

Run with --help for the full flag descriptions and defaults.
"""

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

try:
    import numpy as np
except ImportError:
    np = None

# Any "<family>_step_<n>.pth" checkpoint, whatever the family is called. Deliberately not a
# fixed (policy|vecnorm) alternation: each trainer saves a different SET of files at the same
# step, and a hardcoded list silently leaks every family it doesn't name --
#   train_ftr.py   (marv_rl/hfc/mitriakov)  policy, vecnorm
#   train_d3qn.py  (atd3qn)                 policy, vecnorm
#   train_icmd3qn.py (icmd3qn)              policy, vecnorm, icm
#   train_sac.py   (ctrac)                  policy, vecnorm, qvalue, cvae
#   train_creps.py (creps)                  creps_state          <- no policy/vecnorm AT ALL
# so pruning by policy/vecnorm pairs left icm_/qvalue_/cvae_step_* accumulating for the whole
# run and made pruning a complete no-op on CREPS. The prefix is non-greedy so that
# "creps_state_step_12.pth" reads as family "creps_state", step 12.
STEP_RE = re.compile(r"^(?P<family>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*?)_step_(?P<step>\d+)\.pth$")

# Files under weights/ that exist ONLY so a respawn inside the same SLURM job can resume.
# Once the job is over nothing can read them again, and they are large: D3QN/ICM-D3QN write
# replay_buffer.pt at replay_buffer_save_fraction x capacity (ICM's capacity is 8M
# transitions), the same reason training_state.pth was already being removed here.
RESUME_ONLY_WEIGHT_FILES = ("training_state.pth", "replay_buffer.pt", "replay_buffer.pt.tmp")

# C-TRAC's SAC replay buffer, memmapped at the JOB root (shared by every attempt_N) rather
# than under a single attempt -- see train_sac.py's scratch_dir comment. Tens of GB (~21 GB
# at 500k transitions), and no attempt-level pruning can ever reach it.
REPLAY_BUFFER_DIRNAME = "replay_buffer"
STUDY_NAME_RE = re.compile(r"study_name:\s*['\"]?([\w.\-]+)['\"]?")
REPEATS_RE = re.compile(r"--repeats\s+(\d+)")
TARGET_REACHED_RE = re.compile(r"Target number of trials already reached")
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")
CANCELLED_RE = re.compile(r"CANCELLED AT")
SUCCESS_RE = re.compile(
    r"exit status: 0|Job finished\.|Eval finished with exit status: 0|wandb: Synced"
)

NON_RUN_NAMES = {"host_libs", "isaac_cache", "isaac_data", "isaac_logs", "test", "terrain_previews"}
ONESHOT_PREFIXES = {"collect_ctrac_dataset_": "ctrac_dataset", "pretrain_ctrac_cvae_": "ctrac_pretrain"}
FAILURE_LOG_SIZE_THRESHOLD = 5 * 1024
FAILURE_ELAPSED_THRESHOLD = 600  # seconds

AUDIT_LOG_DEFAULT_NAME = "clean_logs_audit.jsonl"
AUDIT_TAIL_MAX_CHARS = 2000

RAW_ACCEL_FILENAMES = ("raw_accel.npz", "raw_accel_eval.npz")
RAW_ACCEL_DEFAULT_BINS = 4096

# Ported from notebooks/slurm_trial_audit.ipynb's classify_trial() so audit-log rows are
# schema-compatible with that notebook's dataframe, even after the source .out/.err are gone.
_RE_EXIT = re.compile(r"finished with exit status:\s*(\d+)")
_RE_HEADER_DT = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\w+\s+\d+\s+\d+:\d+:\d+\s+\d+)", re.MULTILINE)
_RE_PY_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_RE_TRIAL_DONE = re.compile(r"Trial\s+\d+\s+done")
_RE_TRAIN_DONE = re.compile(r"Training finished with exit (?:code|status):?\s*0|Job finished\.", re.IGNORECASE)
_RE_TRIAL_PRUNE = re.compile(r"Trial(?:\s+\d+)?\s+pruned|Trial\s+\d+\s+finished with value")
_RE_CANCELLED_TIMEOUT = re.compile(r"CANCELLED AT .* DUE TO TIME LIMIT")
_RE_SLURM_KILL = re.compile(r"CANCELLED AT")
_RE_TQDM = re.compile(r"\[(\d+):(\d{2}):(\d{2})<")
_RE_ARRAY_TASK = re.compile(r"Array task\s*:\s*(\d+)")
_RE_JOB_ARRAY = re.compile(r"Job array ID\s*:\s*(\d+)")
_RE_CUDA = re.compile(
    r"CUDA error|Training failed.*CUDA|illegal memory access|CUDA.*out of memory", re.IGNORECASE
)
_RE_TRACEBACK_ANCHORED = re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)
_RE_EXCEPTION_TYPE = re.compile(
    r"^([A-Za-z][A-Za-z0-9_.]*(?:Error|Exception|Interrupt|Warning|Killed))\b", re.MULTILINE
)
_EXCEPTION_SKIP = {"Warning", "UserWarning", "FutureWarning", "DeprecationWarning"}


@dataclass
class Stats:
    studies_merged: int = 0
    units_pruned: int = 0
    units_deleted: int = 0
    top_dirs_removed: int = 0
    eval_dirs_deleted: int = 0
    checkpoints_kept: int = 0
    bytes_freed: int = 0
    audit_records: int = 0
    replay_buffers_removed: int = 0
    replay_buffer_bytes_freed: int = 0
    raw_accel_compacted: int = 0
    raw_accel_bytes_freed: int = 0
    raw_accel_corrupted_deleted: int = 0


def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def backup_before_remove(path: Path, logs_dir: Path, backup_root: Path | None, dry_run: bool, log) -> None:
    """Copy path into backup_root (mirroring its position relative to logs_dir) before it gets
    deleted/overwritten for real. No-op if backup_root is None or this is a dry run."""
    if backup_root is None or dry_run:
        return
    rel = path.resolve().relative_to(logs_dir.resolve())
    dest = backup_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"  [backup] {path} -> {dest}")
    if path.is_dir():
        shutil.copytree(path, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(path, dest)


def dir_size(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def mtime_range(path: Path):
    mtimes = []
    for f in path.rglob("*"):
        if f.is_file():
            try:
                mtimes.append(f.stat().st_mtime)
            except OSError:
                pass
    if not mtimes:
        return None, None
    return min(mtimes), max(mtimes)


def is_recent(path: Path, min_age_hours: float, now: float) -> bool:
    _, hi = mtime_range(path)
    if hi is None:
        return False
    return (now - hi) < min_age_hours * 3600


def extract_jobid(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def csv_has_progress(unit_dir: Path) -> bool:
    for csv_path in unit_dir.glob("*.csv"):
        try:
            with open(csv_path, newline="") as fh:
                reader = csv.reader(fh)
                next(reader, None)
                if next(reader, None) is not None:
                    return True
        except OSError:
            continue
    return False


def log_files_for(unit_dir: Path) -> list[Path]:
    patterns = ("*.out", "*.err", "stdout.log", "stderr.log")
    seen = {}
    for pat in patterns:
        for f in unit_dir.glob(pat):
            seen[f] = f
    return list(seen.values())


def looks_like_failure(unit_dir: Path) -> tuple[bool, str]:
    log_files = log_files_for(unit_dir)
    text = "".join(read_text_safe(f) for f in log_files)

    if SUCCESS_RE.search(text) or TARGET_REACHED_RE.search(text):
        return False, "clean completion / no-op marker present in logs"
    if TRACEBACK_RE.search(text):
        return True, "Traceback in logs"
    if CANCELLED_RE.search(text):
        return True, "SLURM CANCELLED signal in logs"

    lo, hi = mtime_range(unit_dir)
    elapsed = (hi - lo) if (lo is not None and hi is not None) else 0
    combined_size = sum(f.stat().st_size for f in log_files if f.exists())
    if not csv_has_progress(unit_dir) and (
        combined_size < FAILURE_LOG_SIZE_THRESHOLD or elapsed < FAILURE_ELAPSED_THRESHOLD
    ):
        return True, f"no CSV progress and short-lived (elapsed={elapsed:.0f}s, log_bytes={combined_size})"
    return False, "no clear failure signal"


def _parse_header_dt(text: str):
    m = _RE_HEADER_DT.search(text)
    if not m:
        return None
    date_str = " ".join(m.group(2).split())
    for fmt in ("%b %d %H:%M:%S %Y", "%b  %d %H:%M:%S %Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _parse_last_py_ts(lines: list[str]):
    for line in reversed(lines):
        m = _RE_PY_TS.match(line)
        if m:
            try:
                return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
    return None


def _parse_max_tqdm_elapsed(text: str):
    max_sec = None
    for m in _RE_TQDM.finditer(text):
        h, mm, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sec = h * 3600 + mm * 60 + s
        if max_sec is None or sec > max_sec:
            max_sec = sec
    return max_sec


def _extract_exception_types(text: str, err_text: str) -> list[str]:
    combined = text + "\n" + err_text
    types = set(_RE_EXCEPTION_TYPE.findall(combined))
    return sorted(types - _EXCEPTION_SKIP)


def classify_unit(unit_dir: Path, job_type: str) -> dict:
    """Classify a run dir exactly like notebooks/slurm_trial_audit.ipynb's classify_trial(),
    so an audit-log row can stand in for the (now-deleted) .out/.err in that notebook's dataframe."""
    out_files = [f for f in log_files_for(unit_dir) if f.suffix == ".out" or f.name == "stdout.log"]
    err_files = [f for f in log_files_for(unit_dir) if f.suffix == ".err" or f.name == "stderr.log"]

    if not out_files:
        return {
            "job_id": None, "task_id": None, "job_type": job_type,
            "exit_code": None, "status": "no_log_found", "hours": None,
            "trial_done": False, "has_traceback": False, "exception_types": "",
            "out_path": None, "log_tail": "",
        }

    out_path = out_files[0]
    text = read_text_safe(out_path)
    lines = text.splitlines()
    err_text = read_text_safe(err_files[0]) if err_files else ""

    exit_m = _RE_EXIT.search(text)
    exit_code = int(exit_m.group(1)) if exit_m else None

    trial_done = bool(_RE_TRIAL_DONE.search(text))
    train_done = bool(_RE_TRAIN_DONE.search(text))
    trial_prune = bool(_RE_TRIAL_PRUNE.search(text) or _RE_TRIAL_PRUNE.search(err_text))
    cuda_crash = bool(_RE_CUDA.search(text) or _RE_CUDA.search(err_text))
    tl_cancel = bool(_RE_CANCELLED_TIMEOUT.search(err_text) or _RE_CANCELLED_TIMEOUT.search(text))
    slurm_cancel = bool(_RE_SLURM_KILL.search(err_text) or _RE_SLURM_KILL.search(text))
    has_traceback = bool(_RE_TRACEBACK_ANCHORED.search(text) or _RE_TRACEBACK_ANCHORED.search(err_text))
    exception_types = _extract_exception_types(text, err_text) if has_traceback else []

    job_array_m = _RE_JOB_ARRAY.search(text)
    task_m = _RE_ARRAY_TASK.search(text)
    job_id = int(job_array_m.group(1)) if job_array_m else None
    task_id = int(task_m.group(1)) if task_m else None

    start_dt = _parse_header_dt(text)
    end_dt = _parse_last_py_ts(lines)
    tqdm_sec = _parse_max_tqdm_elapsed(err_text) or _parse_max_tqdm_elapsed(text)

    hours = None
    if start_dt and end_dt and end_dt > start_dt:
        hours = (end_dt - start_dt).total_seconds() / 3600
    elif tqdm_sec is not None:
        hours = tqdm_sec / 3600

    if trial_done or train_done:
        status = "success"
    elif trial_prune:
        status = "pruned"
    elif cuda_crash:
        status = "crash_cuda"
    elif exit_code == 75:
        status = "crash_75_container_launch"
    elif exit_code == 255:
        status = "crash_255_container_killed"
    elif tl_cancel:
        status = "timeout"
    elif slurm_cancel:
        status = "cancelled_with_error" if has_traceback else "cancelled"
    elif exit_code == 1:
        status = "crash_1_python"
    elif exit_code == 0:
        status = "exit0_no_result"
    elif exit_code is None:
        status = "killed_with_error" if has_traceback else "killed_no_exit"
    else:
        status = f"crash_{exit_code}_unknown"

    return {
        "job_id": job_id,
        "task_id": task_id,
        "job_type": job_type,
        "exit_code": exit_code,
        "status": status,
        "hours": hours,
        "trial_done": trial_done,
        "has_traceback": has_traceback,
        "exception_types": ", ".join(exception_types) if exception_types else "",
        "out_path": str(out_path),
        "log_tail": (err_text or text)[-AUDIT_TAIL_MAX_CHARS:],
    }


def record_deletion(audit_fh, unit_dir: Path, job_type: str, reason: str, bytes_freed: int,
                     dry_run: bool, log, stats: Stats):
    stats.audit_records += 1
    if dry_run:
        log(f"  [audit] would log classification for {unit_dir}")
        return
    record = classify_unit(unit_dir, job_type)
    record["unit_dir"] = str(unit_dir)
    record["deletion_reason"] = reason
    record["bytes_freed"] = bytes_freed
    record["deleted_at"] = datetime.now().isoformat(timespec="seconds")
    if audit_fh is not None:
        audit_fh.write(json.dumps(record, default=str) + "\n")
        audit_fh.flush()


def compact_raw_accel_file(path: Path, bins: int, dry_run: bool, log, stats: Stats) -> None:
    """Replace a raw_accel*.npz's full per-sample "accel" array with a fine-grained histogram
    plus exact summary stats (count/mean/std/min/max), preserving everything
    scripts/plot_shock_distribution.py needs while shrinking the file by orders of magnitude.

    A corrupted/unreadable npz (e.g. truncated by a job killed mid-write) has no salvageable
    data either way -- np.load already can't read it, so nothing downstream could either -- and
    is deleted outright rather than left behind. Already-compacted or empty files are left alone.
    Updates stats in place.
    """
    assert np is not None, "numpy is required for --compact-raw-accel"
    try:
        with np.load(path) as data:
            if "accel" not in data.files:
                return  # already compacted, or an unrecognized schema -- leave it alone
            accel = np.asarray(data["accel"])
    except Exception as e:
        original_size = path.stat().st_size
        log(f"  [raw_accel] DELETE corrupted/unreadable ({e}) {path}")
        if not dry_run:
            path.unlink(missing_ok=True)
        stats.raw_accel_corrupted_deleted += 1
        stats.bytes_freed += original_size
        return

    if accel.size == 0:
        return

    original_size = path.stat().st_size
    counts, edges = np.histogram(accel, bins=bins)
    n = int(accel.size)

    log(f"  [raw_accel] compact {path} ({n:,} samples, {original_size / 1e6:.1f} MB -> {bins}-bin histogram)")
    if dry_run:
        stats.raw_accel_compacted += 1
        stats.raw_accel_bytes_freed += original_size
        stats.bytes_freed += original_size
        return

    # np.savez_compressed force-appends ".npz" if the target doesn't already end with it, so the
    # temp name must itself end in ".npz" or we'd silently write "<name>.tmp.npz" and the rename below
    # would fail to find it.
    tmp_path = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp_path,
        hist_counts=counts.astype(np.int64),
        hist_edges=edges.astype(np.float64),
        count=np.int64(n),
        mean=np.float64(accel.mean()),
        std=np.float64(accel.std()),
        min=np.float64(accel.min()),
        max=np.float64(accel.max()),
    )
    tmp_path.replace(path)
    new_size = path.stat().st_size
    freed = original_size - new_size
    log(f"    -> {new_size / 1e3:.1f} KB ({original_size / max(new_size, 1):.0f}x smaller)")
    stats.raw_accel_compacted += 1
    stats.raw_accel_bytes_freed += freed
    stats.bytes_freed += freed


def compact_raw_accel_files(logs_dir: Path, bins: int, min_age_hours: float, now: float,
                             dry_run: bool, log, stats: Stats):
    for name in RAW_ACCEL_FILENAMES:
        for f in sorted(logs_dir.rglob(name)):
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            if (now - mtime) < min_age_hours * 3600:
                log(f"  [raw_accel] SKIP (modified within {min_age_hours}h, could still be actively "
                    f"appended to) {f}")
                continue
            compact_raw_accel_file(f, bins, dry_run, log, stats)


def parse_step_slots(weights_dir: Path) -> dict[int, dict[str, Path]]:
    """Group every "<family>_step_<n>.pth" under weights_dir by step, as {n: {family: path}}.

    A "slot" is one training step with all the files that step wrote, whichever trainer
    wrote it. Grouping by step rather than intersecting a fixed pair of families is what
    keeps ctrac's qvalue_/cvae_, icmd3qn's icm_ and creps' creps_state_ checkpoints in the
    same accounting as policy_/vecnorm_ -- and it also picks up a half-written slot from a
    crash (policy_step_N with no vecnorm_step_N), which the old pairing dropped on the floor
    and therefore never deleted.
    """
    slots: dict[int, dict[str, Path]] = {}
    for f in weights_dir.glob("*_step_*.pth"):
        m = STEP_RE.match(f.name)
        if m:
            slots.setdefault(int(m.group("step")), {})[m.group("family")] = f
    return slots


def prune_weights(weights_dir: Path, keep_slots: int, logs_dir: Path, backup_root: Path | None,
                   dry_run: bool, log) -> tuple[int, int]:
    slots = parse_step_slots(weights_dir)
    all_ns = sorted(slots)
    target_max = all_ns[-1] if all_ns else 0
    kept = 0
    freed = 0

    # "This run already has a terminal checkpoint set" -- any family, not policy/vecnorm
    # specifically, since CREPS only ever writes creps_state_final.pth. Matching on any
    # rather than all is also what stops a partially-written set (policy_final.pth present,
    # vecnorm_final.pth missing) from being silently overwritten by the rename below.
    has_terminal = any(weights_dir.glob("*_final.pth")) or any(weights_dir.glob("*_crash.pth"))

    if has_terminal:
        kept += 1
    elif all_ns:
        max_n = all_ns[-1]
        slot = slots.pop(max_n)
        # Rename EVERY family saved at that step. Renaming only policy/vecnorm would strand
        # the qvalue_/cvae_ (ctrac) or icm_ (icmd3qn) files that belong to the same
        # checkpoint, leaving them to be deleted below as an ordinary step slot.
        names = " + ".join(sorted(p.name for p in slot.values()))
        log(f"  rename {names} -> *_final.pth in {weights_dir}")
        if not dry_run:
            for family, path in slot.items():
                path.rename(weights_dir / f"{family}_final.pth")
        kept += 1

    extra_slots = keep_slots - 1
    remaining_ns = sorted(slots)
    chosen = set()
    if extra_slots > 0 and remaining_ns:
        # Only slots that still carry the run's primary checkpoint are worth keeping. An
        # earlier cleanup that understood policy/vecnorm but not qvalue/cvae/icm left orphan
        # slots behind holding just the secondary families, and picking one of those as the
        # 1/3 or 2/3 keep preserves a step that nothing can be evaluated from. Fall back to
        # the full pool if no slot has the primary family (nothing better to choose from).
        families = set().union(*slots.values())
        primary = "policy" if "policy" in families else (sorted(families)[0] if len(families) == 1 else None)
        candidates = [n for n in remaining_ns if primary in slots[n]] if primary else remaining_ns
        candidates = candidates or remaining_ns
        for frac in (1 / 3, 2 / 3):
            target = target_max * frac
            best = min(candidates, key=lambda n: abs(n - target))
            chosen.add(best)

    for n in remaining_ns:
        slot = slots[n]
        if n in chosen:
            kept += 1
            continue
        for _family, path in sorted(slot.items()):
            try:
                sz = path.stat().st_size
            except OSError:
                continue
            log(f"  delete {path.name} in {weights_dir}")
            backup_before_remove(path, logs_dir, backup_root, dry_run, log)
            if not dry_run:
                path.unlink(missing_ok=True)
            freed += sz

    for fname in RESUME_ONLY_WEIGHT_FILES:
        f = weights_dir / fname
        if not f.exists():
            continue
        sz = f.stat().st_size
        log(f"  delete {fname} in {weights_dir}")
        backup_before_remove(f, logs_dir, backup_root, dry_run, log)
        if not dry_run:
            f.unlink()
        freed += sz

    return kept, freed


def remove_stale_replay_buffer(top_dir: Path, min_age_hours: float, now: float, logs_dir: Path,
                                backup_root: Path | None, dry_run: bool, log, stats: Stats):
    """Delete <job_dir>/replay_buffer/ once the job that owned it is over.

    This is C-TRAC's SAC buffer (train_sac.py): memmapped at the RUN root so that every
    attempt_N of one SLURM job shares it, which is exactly why per-attempt pruning cannot
    see it. It is tens of GB (~21 GB at 500k transitions x history_len 16) and its only
    consumer is a respawn *within the same job id* -- once that job has exited, nothing can
    ever read it again.

    Guarded on the whole job dir being untouched for --min-age-hours rather than on
    --keep-last: keep-last protects checkpoints for later analysis, whereas this is pure
    resume state, and a finished run's newest attempt is usually keep-last-protected.
    """
    rb = top_dir / REPLAY_BUFFER_DIRNAME
    if not rb.is_dir():
        return
    if is_recent(top_dir, min_age_hours, now):
        log(f"SKIP replay buffer (job modified within {min_age_hours}h — may still be running) {rb}")
        return
    sz = dir_size(rb)
    log(f"DELETE replay buffer ({sz / 1e9:.2f} GB of resume-only state, job has finished) {rb}")
    backup_before_remove(rb, logs_dir, backup_root, dry_run, log)
    if not dry_run:
        shutil.rmtree(rb, ignore_errors=True)
    stats.replay_buffers_removed += 1
    stats.replay_buffer_bytes_freed += sz
    stats.bytes_freed += sz


def train_units(top_dir: Path) -> list[Path]:
    attempts = sorted(d for d in top_dir.iterdir() if d.is_dir() and re.match(r"attempt_\d+$", d.name))
    return attempts if attempts else [top_dir]


def recover_units(top_dir: Path) -> list[Path]:
    subdirs = sorted(d for d in top_dir.iterdir() if d.is_dir())
    return subdirs if subdirs else [top_dir]


def optuna_trial_units(top_dir: Path) -> list[Path]:
    return sorted(d for d in top_dir.iterdir() if d.is_dir())


def read_study_name(top_dir: Path) -> str | None:
    for yml in top_dir.glob("*.yaml"):
        text = read_text_safe(yml)
        if yaml is not None:
            try:
                data = yaml.safe_load(text)
                if isinstance(data, dict) and isinstance(data.get("study_name"), str):
                    return data["study_name"]
            except Exception:
                pass
        m = STUDY_NAME_RE.search(text)
        if m:
            return m.group(1)
    return None


def process_unit_group(top_dir: Path, units: list[Path], keep_slots: int, job_type: str, log_scope: str,
                        protected: set, min_age_hours: float, now: float, logs_dir: Path,
                        backup_root: Path | None, dry_run: bool, log, stats: Stats, audit_fh):
    """log_scope: 'per_unit' if each unit has its own out/err (optuna trials, recover tasks),
    'top_dir_only' if out/err are shared across units and only live at top_dir (train attempts)."""
    any_survived = False
    top_dir_removed = False

    for u in units:
        if u in protected:
            log(f"SKIP (protected by --keep-last) {u}")
            any_survived = True
            continue
        if is_recent(u, min_age_hours, now):
            log(f"SKIP (modified within {min_age_hours}h) {u}")
            any_survived = True
            continue

        weights_dir = u / "weights"
        has_weights = weights_dir.is_dir() and any(weights_dir.iterdir())

        if not has_weights:
            failed, reason = looks_like_failure(u)
            if failed:
                sz = dir_size(u)
                if log_scope == "per_unit" or u == top_dir:
                    record_deletion(audit_fh, u, job_type, reason, sz, dry_run, log, stats)
                log(f"DELETE (failed: {reason}) {u}")
                backup_before_remove(u, logs_dir, backup_root, dry_run, log)
                if not dry_run:
                    shutil.rmtree(u, ignore_errors=True)
                stats.units_deleted += 1
                stats.bytes_freed += sz
                if u == top_dir:
                    top_dir_removed = True
                continue
            log(f"KEEP (no weights, no failure signal) {u}")
            any_survived = True
            continue

        log(f"PRUNE weights in {u}")
        kept, freed = prune_weights(weights_dir, keep_slots, logs_dir, backup_root, dry_run, log)
        stats.units_pruned += 1
        stats.checkpoints_kept += kept
        stats.bytes_freed += freed
        any_survived = True

    if not units:
        log(f"SKIP {top_dir} (no recognized sub-units -- e.g. legacy flat trial-log layout "
            f"with no per-trial subdirs or weights; left untouched)")
        return

    if not any_survived and not top_dir_removed and top_dir.exists():
        if log_scope == "top_dir_only":
            sz = dir_size(top_dir)
            record_deletion(audit_fh, top_dir, job_type, "all sub-units removed", sz, dry_run, log, stats)
        log(f"REMOVE now-empty top-level dir {top_dir}")
        backup_before_remove(top_dir, logs_dir, backup_root, dry_run, log)
        if not dry_run:
            shutil.rmtree(top_dir, ignore_errors=True)
        stats.top_dirs_removed += 1


def merge_optuna_studies(optuna_top: list[Path], min_age_hours: float, now: float,
                          dry_run: bool, log, stats: Stats):
    study_map: dict[str, list[Path]] = {}
    for d in optuna_top:
        study = read_study_name(d) or f"__unknown__{d.name}"
        study_map.setdefault(study, []).append(d)

    for study, dirs in study_map.items():
        if len(dirs) <= 1:
            continue
        if any(is_recent(d, min_age_hours, now) for d in dirs):
            log(f"SKIP merge of study '{study}' ({len(dirs)} dirs) -- one or more modified within {min_age_hours}h")
            continue

        dirs_sorted = sorted(dirs, key=lambda d: extract_jobid(d.name))
        canonical = dirs_sorted[0]
        log(f"MERGE study '{study}': {[d.name for d in dirs_sorted[1:]]} -> {canonical.name}")

        for donor in dirs_sorted[1:]:
            moved_all = True
            for trial_dir in optuna_trial_units(donor):
                dest = canonical / trial_dir.name
                if dest.exists():
                    log(f"  WARN collision moving {trial_dir} -> {dest}, leaving in place")
                    moved_all = False
                    continue
                log(f"  move {trial_dir} -> {dest}")
                if not dry_run:
                    shutil.move(str(trial_dir), str(dest))
            if moved_all:
                log(f"  remove donor study dir {donor}")
                if not dry_run:
                    shutil.rmtree(donor, ignore_errors=True)
        stats.studies_merged += 1


def process_eval_dir(top_dir: Path, min_repeats: int, min_age_hours: float, now: float, logs_dir: Path,
                      backup_root: Path | None, dry_run: bool, log, stats: Stats, audit_fh):
    if is_recent(top_dir, min_age_hours, now):
        log(f"SKIP eval (modified within {min_age_hours}h) {top_dir}")
        return

    summary = top_dir / "eval_summary.csv"
    if not summary.is_file():
        # Many eval entry points (eval_auto.sh sweeps, eval_creps.sh, eval_d3qn.sh runs without
        # --output_dir) write their results straight to an experiments/... dir instead of
        # <log_dir>/eval_summary.csv -- a missing summary here is not itself evidence of a crash.
        # Fall back to the same log-text success/failure heuristic used for training units.
        failed, reason = looks_like_failure(top_dir)
        if not failed:
            log(f"KEEP eval (no eval_summary.csv, but no failure signal -- {reason}) {top_dir}")
            return
        sz = dir_size(top_dir)
        record_deletion(audit_fh, top_dir, "eval", reason, sz, dry_run, log, stats)
        log(f"DELETE eval dir ({reason}) {top_dir}")
        backup_before_remove(top_dir, logs_dir, backup_root, dry_run, log)
        if not dry_run:
            shutil.rmtree(top_dir, ignore_errors=True)
        stats.eval_dirs_deleted += 1
        stats.bytes_freed += sz
        return

    completed = 0
    try:
        with open(summary, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    completed = max(completed, int(row.get("repeat", 0)))
                except (TypeError, ValueError):
                    continue
    except OSError:
        pass

    intended = completed
    for out_file in log_files_for(top_dir):
        m = REPEATS_RE.search(read_text_safe(out_file))
        if m:
            intended = int(m.group(1))
            break

    if completed >= min_repeats:
        log(f"KEEP eval (completed {completed} repeats) {top_dir}")
        return
    if intended < min_repeats:
        log(f"KEEP eval (short by design: intended {intended} repeats) {top_dir}")
        return

    sz = dir_size(top_dir)
    reason = f"incomplete: {completed}/{intended} repeats"
    record_deletion(audit_fh, top_dir, "eval", reason, sz, dry_run, log, stats)
    log(f"DELETE eval dir ({reason}) {top_dir}")
    backup_before_remove(top_dir, logs_dir, backup_root, dry_run, log)
    if not dry_run:
        shutil.rmtree(top_dir, ignore_errors=True)
    stats.eval_dirs_deleted += 1
    stats.bytes_freed += sz


def print_summary(stats: Stats, dry_run: bool, audit_path: Path):
    mode = "DRY RUN -- no changes were made" if dry_run else "changes applied"
    gb = stats.bytes_freed / 1e9
    print("\n" + "=" * 60)
    print(f"clean_logs summary ({mode})")
    print(f"  optuna studies merged:        {stats.studies_merged}")
    print(f"  training/optuna units pruned: {stats.units_pruned}  (checkpoint pairs kept: {stats.checkpoints_kept})")
    print(f"  failed units deleted:         {stats.units_deleted}")
    print(f"  now-empty top dirs removed:   {stats.top_dirs_removed}")
    print(f"  incomplete eval dirs deleted: {stats.eval_dirs_deleted}")
    if stats.replay_buffers_removed:
        verb = "would be removed" if dry_run else "removed"
        print(f"  job replay buffers {verb}:   {stats.replay_buffers_removed}  "
              f"(~{stats.replay_buffer_bytes_freed / 1e9:.2f} GB)")
    if stats.raw_accel_compacted:
        verb = "would be compacted" if dry_run else "compacted"
        print(f"  raw_accel*.npz {verb}:  {stats.raw_accel_compacted}  "
              f"(~{stats.raw_accel_bytes_freed / 1e9:.2f} GB{' est.' if dry_run else ''})")
    if stats.raw_accel_corrupted_deleted:
        verb = "would be deleted" if dry_run else "deleted"
        print(f"  corrupted raw_accel*.npz {verb} (unreadable, no salvageable data): "
              f"{stats.raw_accel_corrupted_deleted}")
    print(f"  space reclaimed:              {gb:.2f} GB")
    if dry_run:
        print(f"  audit records that would be appended to {audit_path}: {stats.audit_records}")
        print("  this was a dry run -- re-run without --dry-run to apply")
    else:
        print(f"  audit records appended to {audit_path}: {stats.audit_records}")
        print("  (pass --dry-run next time to preview before making changes)")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Clean up SLURM training/optuna/eval log dirs: prune old checkpoints, merge optuna "
            "studies split across SLURM array resubmissions, and delete failed or incomplete runs."
        )
    )
    p.add_argument("--logs-dir", default="logs", help="path to the logs/ directory to clean (default: ./logs)")
    p.add_argument("--dry-run", action="store_true", help="print planned actions without making any changes")
    p.add_argument(
        "--keep-last", type=int, default=5, metavar="N",
        help="protect the N most-recently-modified training attempts + optuna trial dirs "
             "(pooled together) from any pruning/deletion (default: 5)",
    )
    p.add_argument(
        "--min-age-hours", type=float, default=12.0, metavar="H",
        help="skip anything whose newest file was modified within H hours, to avoid touching "
             "an actively-running job (default: 12)",
    )
    p.add_argument(
        "--eval-min-repeats", type=int, default=5, metavar="R",
        help="eval dirs with fewer completed repeats than R are deleted, unless the eval was only "
             "ever configured for fewer than R repeats (default: 5)",
    )
    p.add_argument(
        "--audit-log", default=None, metavar="PATH",
        help="JSONL file to append one classification record to for every deleted run, before its "
             "logs are gone, so notebooks/slurm_trial_audit.ipynb can still be used in full "
             f"(default: <logs-dir>/{AUDIT_LOG_DEFAULT_NAME}). Never written to during --dry-run.",
    )
    p.add_argument(
        "--compact-raw-accel", action="store_true",
        help="replace raw_accel.npz/raw_accel_eval.npz full per-sample dumps (up to ~200MB each) with "
             "a fine-grained histogram + exact summary stats (a few KB), preserving what "
             "scripts/plot_shock_distribution.py needs. Off by default. Requires numpy. Skips anything "
             "within --min-age-hours to avoid racing a job still appending to the file.",
    )
    p.add_argument(
        "--raw-accel-bins", type=int, default=RAW_ACCEL_DEFAULT_BINS, metavar="N",
        help=f"histogram bin count for --compact-raw-accel (default: {RAW_ACCEL_DEFAULT_BINS})",
    )
    p.add_argument(
        "--keep-replay-buffers", action="store_true",
        help="do not delete <job_dir>/replay_buffer/ (C-TRAC's memmapped SAC buffer, ~21 GB per "
             "job). By default a job untouched for --min-age-hours has it removed: it is "
             "resume-only state that only a respawn within that same SLURM job could ever read. "
             "D3QN/ICM-D3QN's per-attempt weights/replay_buffer.pt is removed with the rest of "
             "the resume state and is not covered by this flag.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="print per-directory reasoning, not just the summary")
    p.add_argument(
        "--exclude", action="append", default=[], metavar="NAME",
        help="top-level dir name under --logs-dir to leave completely untouched (repeatable). "
             "Matched against the exact directory name (e.g. --exclude train_ctrac_11311704).",
    )
    p.add_argument(
        "--backup-dir", default=None, metavar="PATH",
        help="before deleting or pruning anything for real, copy it into PATH first (mirroring its "
             "path relative to --logs-dir). No-op during --dry-run. Not cleaned up automatically -- "
             "remove it yourself once you've verified the run.",
    )
    return p


def main():
    args = build_parser().parse_args()

    logs_dir = Path(args.logs_dir).resolve()
    if not logs_dir.is_dir():
        print(f"error: {logs_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    if args.compact_raw_accel and np is None:
        print("error: --compact-raw-accel requires numpy (try /home/robot/conda/envs/isaaclab/bin/python)",
              file=sys.stderr)
        sys.exit(1)

    now = datetime.now().timestamp()
    stats = Stats()
    audit_path = Path(args.audit_log).resolve() if args.audit_log else logs_dir / AUDIT_LOG_DEFAULT_NAME
    exclude_names = set(args.exclude)
    backup_root = Path(args.backup_dir).resolve() if args.backup_dir else None
    if backup_root is not None and not args.dry_run:
        backup_root.mkdir(parents=True, exist_ok=True)

    def log(msg):
        if args.verbose:
            print(msg)

    train_top, optuna_top, recover_top, eval_top, oneshot_top = [], [], [], [], []
    for d in sorted(logs_dir.iterdir()):
        if d.is_symlink():
            # train_auto.sbatch renames the job to train_<module_name> once it has read
            # module_name out of the config, and leaves logs/train_auto_<job_id> behind as a
            # symlink to logs/train_<module>_<job_id>. Both names show up in this listing, so
            # following the link would process the very same attempt dirs twice: two audit
            # rows and two backup copytrees per unit, double-counted bytes_freed, and a bogus
            # "removed top-level dir" (shutil.rmtree refuses to delete a symlink, and
            # ignore_errors=True swallows the refusal). The real directory is handled on its
            # own entry.
            print(f"SKIP (symlink -> {d.readlink()}) {d}")
            continue
        if not d.is_dir():
            continue
        name = d.name
        if name in exclude_names:
            print(f"SKIP (--exclude) {d}")
            continue
        matched_prefix = next((p for p in ONESHOT_PREFIXES if name.startswith(p)), None)
        if name.startswith("recover_optuna_"):
            recover_top.append(d)
        elif name.startswith("optuna_"):
            optuna_top.append(d)
        elif name.startswith("train_") or name.startswith("long_train_"):
            train_top.append(d)
        elif name.startswith("eval_"):
            eval_top.append(d)
        elif matched_prefix is not None:
            oneshot_top.append((d, ONESHOT_PREFIXES[matched_prefix]))
        elif name in NON_RUN_NAMES or name.startswith("wandb"):
            continue
        else:
            print(f"WARN: unrecognized dir '{name}' left untouched (not matched to any known category)")

    merge_optuna_studies(optuna_top, args.min_age_hours, now, args.dry_run, log, stats)
    optuna_top = sorted(d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("optuna_"))

    pool = []
    for d in train_top:
        pool.extend(train_units(d))
    for d in recover_top:
        pool.extend(recover_units(d))
    for d in optuna_top:
        pool.extend(optuna_trial_units(d))
    for d, _job_type in oneshot_top:
        pool.append(d)

    def unit_key(u: Path):
        _, hi = mtime_range(u)
        return hi if hi is not None else 0.0

    protected = set(sorted(pool, key=unit_key, reverse=True)[: args.keep_last])

    audit_fh = None if args.dry_run else open(audit_path, "a", encoding="utf-8")
    try:
        for d in train_top:
            # Before the units, so the buffer's bytes are attributed here rather than being
            # counted again inside dir_size(top_dir) if the whole run then gets removed.
            if not args.keep_replay_buffers:
                remove_stale_replay_buffer(d, args.min_age_hours, now, logs_dir, backup_root,
                                           args.dry_run, log, stats)
            process_unit_group(d, train_units(d), 3, "train", "top_dir_only", protected,
                                args.min_age_hours, now, logs_dir, backup_root, args.dry_run, log, stats, audit_fh)
        for d in recover_top:
            process_unit_group(d, recover_units(d), 3, "recover_optuna", "per_unit", protected,
                                args.min_age_hours, now, logs_dir, backup_root, args.dry_run, log, stats, audit_fh)
        for d in optuna_top:
            process_unit_group(d, optuna_trial_units(d), 1, "optuna", "per_unit", protected,
                                args.min_age_hours, now, logs_dir, backup_root, args.dry_run, log, stats, audit_fh)
        for d, job_type in oneshot_top:
            # One-shot jobs (dataset collection, C-VAE pretraining): a single unrestructured
            # dir with no attempt_N/ and no weights/ -- their deliverable (a dataset dir or a
            # single checkpoint file) sits directly at the top, so there's nothing to prune;
            # process_unit_group's "no weights" branch reduces to a pure log-based success/
            # failure check, which is exactly what's needed here.
            process_unit_group(d, [d], 1, job_type, "per_unit", protected,
                                args.min_age_hours, now, logs_dir, backup_root, args.dry_run, log, stats, audit_fh)
        for d in eval_top:
            process_eval_dir(d, args.eval_min_repeats, args.min_age_hours, now, logs_dir, backup_root,
                              args.dry_run, log, stats, audit_fh)
    finally:
        if audit_fh is not None:
            audit_fh.close()

    if args.compact_raw_accel:
        compact_raw_accel_files(logs_dir, args.raw_accel_bins, args.min_age_hours, now, args.dry_run, log, stats)

    print_summary(stats, args.dry_run, audit_path)


if __name__ == "__main__":
    main()
