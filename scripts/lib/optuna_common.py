"""Helpers shared by the Optuna post-processing scripts in scripts/.

deploy_best_config.py and deploy_gp_optimal.py both turn a finished study into a versioned
training config; optuna_best_params.py prints a study's best parameters. The pieces they had
each copied live here. Import with::

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lib.optuna_common import ROOT, latest_study_name, next_version_path, get_storage
"""

import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The config every Optuna study in this workspace has searched from.
DEFAULT_BASE_CONFIG = ROOT / "configs" / "thesis_main" / "ftr_config_A_new.yaml"
DEFAULT_SQLITE_DB = ROOT / "optuna" / "optuna.db"


def latest_study_name(db_path: Path) -> str:
    """Name of the most recently created study in a SQLite Optuna DB."""
    con = sqlite3.connect(db_path)
    row = con.execute("SELECT study_name FROM studies ORDER BY study_id DESC LIMIT 1").fetchone()
    con.close()
    if row is None:
        raise RuntimeError(f"No studies found in {db_path}")
    return row[0]


def next_version_path(configs_dir: Path, stem: str) -> Path:
    """``<configs_dir>/<stem>_v<N>.yaml`` with N one past the highest existing version."""
    nums = [int(m.group(1)) for p in configs_dir.glob(f"{stem}_v*.yaml")
            if (m := re.search(r"_v(\d+)$", p.stem))]
    return configs_dir / f"{stem}_v{(max(nums) + 1) if nums else 1}.yaml"


def get_storage():
    """Optuna RDBStorage from ``<workspace>/optuna_db.yaml`` — a ``url:`` or PostgreSQL fields.

    Same file and same precedence as marv_rl_training.training.optuna_train_ftr, so the
    analysis scripts always talk to the DB the trials were written to.
    """
    import optuna
    from omegaconf import OmegaConf

    db_path = ROOT / "optuna_db.yaml"
    if not db_path.exists():
        print(f"ERROR: {db_path} not found.", file=sys.stderr)
        sys.exit(1)
    db = OmegaConf.load(db_path)
    if "url" in db:
        conn_str = db["url"]
    else:
        sslmode = db.get("sslmode", "require")
        conn_str = (f"postgresql+psycopg2://{db['db_user']}:{db['db_password']}"
                    f"@{db['db_host']}:{db['db_port']}/{db['db_name']}?sslmode={sslmode}")
    return optuna.storages.RDBStorage(conn_str)
