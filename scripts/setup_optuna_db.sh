#!/usr/bin/env bash
# setup_optuna_db.sh — initialise a local PostgreSQL instance for Optuna hyperparameter search.
#
# Run once on the head/login node before submitting SLURM array jobs.
# The resulting DB is accessible from all compute nodes via the cluster's internal network.
#
# Usage:
#   bash scripts/setup_optuna_db.sh            # use defaults
#   bash scripts/setup_optuna_db.sh --port 5433 --data-dir /scratch/pg_data
#
# After running, copy the printed optuna_db.yaml block into
#   optuna_db.yaml  (workspace root)
# or pass --write-yaml to overwrite it automatically.

set -euo pipefail

# ── load PostgreSQL module (EasyBuild cluster) ────────────────────────────────
PG_MODULE="PostgreSQL/17.5-GCCcore-14.3.0"
if command -v module &>/dev/null; then
    module load "$PG_MODULE" 2>/dev/null \
        && echo "Loaded module: $PG_MODULE" \
        || echo "Warning: could not load $PG_MODULE — will search PATH anyway"
fi

# ── defaults (override via CLI flags) ────────────────────────────────────────
DB_NAME="optuna_db"
DB_USER="optuna"
DB_PASSWORD=""          # generated randomly if empty
DB_PORT=5432
DATA_DIR="${HOME}/postgres_optuna"
WRITE_YAML=0
YAML_PATH=""            # resolved below if not set

# ── CLI parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --db-name)    DB_NAME="$2";    shift 2 ;;
        --db-user)    DB_USER="$2";    shift 2 ;;
        --password)   DB_PASSWORD="$2"; shift 2 ;;
        --port)       DB_PORT="$2";    shift 2 ;;
        --data-dir)   DATA_DIR="$2";   shift 2 ;;
        --write-yaml) WRITE_YAML=1;    shift   ;;
        --yaml-path)  YAML_PATH="$2";  shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── resolve YAML path ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(dirname "$SCRIPT_DIR")"
if [[ -z "$YAML_PATH" ]]; then
    YAML_PATH="${WS_ROOT}/optuna_db.yaml"
fi

# ── find postgres binaries ────────────────────────────────────────────────────
find_bin() {
    local bin="$1"
    # 1. already on PATH
    if command -v "$bin" &>/dev/null; then
        echo "$(command -v "$bin")"; return
    fi
    # 2. common HPC module locations
    for candidate in \
        /usr/lib/postgresql/*/bin/"$bin" \
        /usr/pgsql-*/bin/"$bin" \
        /opt/homebrew/bin/"$bin" \
        /usr/local/bin/"$bin"; do
        # shellcheck disable=SC2086
        for match in $candidate; do
            [[ -x "$match" ]] && { echo "$match"; return; }
        done
    done
    echo ""
}

INITDB=$(find_bin initdb)
PG_CTL=$(find_bin pg_ctl)
PSQL=$(find_bin psql)
CREATEDB=$(find_bin createdb)

for bin_var in INITDB PG_CTL PSQL CREATEDB; do
    val="${!bin_var}"
    if [[ -z "$val" ]]; then
        echo "ERROR: could not find '${bin_var,,}'. Install postgresql-server or load the module."
        echo "       On SLURM clusters: 'module load postgresql' or 'module load postgresql-server'"
        exit 1
    fi
    echo "  found ${bin_var,,}: $val"
done

# ── generate password if not provided ─────────────────────────────────────────
if [[ -z "$DB_PASSWORD" ]]; then
    DB_PASSWORD="$(head -c 18 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 20)"
fi

# ── host detection ────────────────────────────────────────────────────────────
DB_HOST="$(hostname -f 2>/dev/null || hostname)"

# ── initdb ───────────────────────────────────────────────────────────────────
if [[ -d "${DATA_DIR}/global" ]]; then
    echo "Data directory ${DATA_DIR} already initialised — skipping initdb."
else
    echo "Initialising PostgreSQL data directory at ${DATA_DIR} ..."
    mkdir -p "$DATA_DIR"
    "$INITDB" -D "$DATA_DIR" --auth=md5 --username=postgres --pwprompt 2>&1 | grep -v "^$" || true
fi

# ── configure port and listen addresses ──────────────────────────────────────
PG_CONF="${DATA_DIR}/postgresql.conf"
# Set port (idempotent: replace existing line or append)
if grep -q "^port\s*=" "$PG_CONF" 2>/dev/null; then
    sed -i "s/^port\s*=.*/port = ${DB_PORT}/" "$PG_CONF"
else
    echo "port = ${DB_PORT}" >> "$PG_CONF"
fi
# Listen on all interfaces so compute nodes can reach the head node
if grep -q "^listen_addresses\s*=" "$PG_CONF" 2>/dev/null; then
    sed -i "s/^listen_addresses\s*=.*/listen_addresses = '*'/" "$PG_CONF"
else
    echo "listen_addresses = '*'" >> "$PG_CONF"
fi

# ── pg_hba.conf: allow password auth from internal network ────────────────────
PG_HBA="${DATA_DIR}/pg_hba.conf"
HBA_LINE="host    ${DB_NAME}    ${DB_USER}    0.0.0.0/0    md5"
if ! grep -qF "$HBA_LINE" "$PG_HBA" 2>/dev/null; then
    echo "$HBA_LINE" >> "$PG_HBA"
    echo "Added pg_hba.conf rule for ${DB_USER}@${DB_NAME} from any host."
fi

# ── start server (skip if already running) ────────────────────────────────────
LOG_FILE="${DATA_DIR}/postgres.log"
if "$PG_CTL" -D "$DATA_DIR" status &>/dev/null; then
    echo "PostgreSQL is already running."
else
    echo "Starting PostgreSQL ..."
    "$PG_CTL" -D "$DATA_DIR" -l "$LOG_FILE" start
    sleep 2
fi

# ── create role and database ──────────────────────────────────────────────────
PG_SUPERUSER="postgres"
PSQL_SUPER="$PSQL -U $PG_SUPERUSER -p $DB_PORT"

# create role (ignore if already exists)
$PSQL_SUPER -d postgres -tc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" \
    | grep -q 1 \
    || $PSQL_SUPER -d postgres -c \
        "CREATE ROLE ${DB_USER} WITH LOGIN PASSWORD '${DB_PASSWORD}';"

# update password in case it changed
$PSQL_SUPER -d postgres -c "ALTER ROLE ${DB_USER} WITH PASSWORD '${DB_PASSWORD}';" &>/dev/null

# create database (ignore if already exists)
$PSQL_SUPER -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" \
    | grep -q 1 \
    || "$CREATEDB" -U "$PG_SUPERUSER" -p "$DB_PORT" -O "$DB_USER" "$DB_NAME"

# grant all on database
$PSQL_SUPER -d postgres -c "GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};" &>/dev/null

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " PostgreSQL ready"
echo "   host     : ${DB_HOST}"
echo "   port     : ${DB_PORT}"
echo "   database : ${DB_NAME}"
echo "   user     : ${DB_USER}"
echo "   password : ${DB_PASSWORD}"
echo "   data dir : ${DATA_DIR}"
echo "   log      : ${LOG_FILE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── write optuna_db.yaml ──────────────────────────────────────────────────────
YAML_CONTENT="db_user: ${DB_USER}
db_password: ${DB_PASSWORD}
db_host: ${DB_HOST}
db_port: ${DB_PORT}
db_name: ${DB_NAME}
sslmode: disable
"

echo ""
echo "optuna_db.yaml content:"
echo "---"
echo "$YAML_CONTENT"
echo "---"

if [[ "$WRITE_YAML" -eq 1 ]]; then
    echo "$YAML_CONTENT" > "$YAML_PATH"
    echo "Written to ${YAML_PATH}"
else
    echo "Pass --write-yaml to overwrite ${YAML_PATH} automatically."
fi

# ── hint: how to start the server after a reboot ─────────────────────────────
echo ""
echo "To restart after a reboot:"
echo "  ${PG_CTL} -D ${DATA_DIR} -l ${LOG_FILE} start"
echo ""
echo "To stop:"
echo "  ${PG_CTL} -D ${DATA_DIR} stop"
