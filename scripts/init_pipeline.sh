#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# SentiSense — Full Pipeline Initialization Script (Ubuntu)
# ═══════════════════════════════════════════════════════════════════════
#
# This script bootstraps the entire SentiSense data pipeline from zero:
#   1. Installs system dependencies (Python, uv)
#   2. Installs Python packages via uv for all modules
#   3. Installs Playwright browser for the scraper
#   4. Sets up PostgreSQL (Docker or native — auto-detected)
#   5. Updates data.csv with newly scraped headlines
#   6. Migrates the full dataset into PostgreSQL
#
# PostgreSQL mode (auto-detected):
#   - If Docker daemon is available → uses docker compose
#   - If no Docker daemon (e.g. inside a container) → installs PostgreSQL
#     natively via apt-get and configures it directly
#
# Usage:
#   chmod +x scripts/init_pipeline.sh
#   ./scripts/init_pipeline.sh
#
# Flags:
#   --skip-scrape    Skip the data.csv update (scraper) step
#   --skip-migrate   Skip the CSV-to-DB migration step
#   --skip-db        Skip PostgreSQL setup entirely
#   --dry-run        Show what would happen without making changes
#
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SKIP_SCRAPE=false
SKIP_MIGRATE=false
SKIP_DB=false
DRY_RUN=false

DB_NAME="sentisense"
DB_USER="sentisense"
DB_PASS="sentisense_dev"

for arg in "$@"; do
    case "$arg" in
        --skip-scrape)  SKIP_SCRAPE=true ;;
        --skip-migrate) SKIP_MIGRATE=true ;;
        --skip-db)      SKIP_DB=true ;;
        --skip-docker)  SKIP_DB=true ;;  # backward compat alias
        --dry-run)      DRY_RUN=true ;;
        *)              echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

step() { echo -e "\n${BLUE}══════════════════════════════════════════════════${NC}"; echo -e "${GREEN}  ▶ $1${NC}"; echo -e "${BLUE}══════════════════════════════════════════════════${NC}\n"; }
info() { echo -e "  ${YELLOW}ℹ${NC}  $1"; }
ok()   { echo -e "  ${GREEN}✓${NC}  $1"; }
fail() { echo -e "  ${RED}✗${NC}  $1"; }

check_cmd() {
    if command -v "$1" &>/dev/null; then
        ok "$1 found: $(command -v "$1")"
        return 0
    else
        return 1
    fi
}

# Run a psql command using the correct method (Docker or native)
run_psql() {
    if [ "$PG_MODE" = "docker" ]; then
        docker exec sentisense-postgres psql -U "$DB_USER" -d "$DB_NAME" "$@"
    else
        PGPASSWORD="$DB_PASS" psql -U "$DB_USER" -d "$DB_NAME" -h localhost "$@"
    fi
}

cd "$PROJECT_ROOT"

# ─────────────────────────────────────────────────────────────────────
# Step 1: System dependencies
# ─────────────────────────────────────────────────────────────────────

step "Step 1/6 — Checking system dependencies"

# Detect Docker availability (used later in Step 4)
DOCKER_AVAILABLE=false
if docker info &>/dev/null 2>&1; then
    DOCKER_AVAILABLE=true
    ok "Docker daemon available: $(docker --version 2>&1 | head -1)"
    if docker compose version &>/dev/null; then
        ok "docker compose v2: $(docker compose version --short)"
    fi
else
    info "Docker daemon not available — will use native PostgreSQL in Step 4"
fi

# Python 3.12+
# On Ubuntu, python3 may point to an older version even when python3.12 is installed.
# Try python3 first, then fall back to python3.12, python3.13, etc.
PYTHON=""
for candidate in python3 python3.12 python3.13 python3.14; do
    if command -v "$candidate" &>/dev/null; then
        if "$candidate" -c "import sys; exit(0 if sys.version_info >= (3,12) else 1)" 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -n "$PYTHON" ]; then
    PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    ok "Python $PY_VERSION (>= 3.12) via $(command -v "$PYTHON")"
else
    fail "Python 3.12+ not found"
    info "Install with: apt install python3.12 python3.12-venv"
    exit 1
fi

# uv
if check_cmd uv; then
    :
else
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ok "uv installed"
fi

# ─────────────────────────────────────────────────────────────────────
# Step 2: Python packages (all via uv)
# ─────────────────────────────────────────────────────────────────────

step "Step 2/6 — Installing Python packages (uv)"

# processing_engine (includes psycopg, loguru, langchain, etc.)
info "Syncing processing_engine..."
cd "$PROJECT_ROOT/processing_engine"
uv sync
ok "processing_engine synced ($(wc -l < uv.lock | tr -d ' ') lines in lockfile)"

# mivzakim_scraper
info "Syncing mivzakim_scraper..."
cd "$PROJECT_ROOT/mivzakim_scraper"
uv sync
ok "mivzakim_scraper synced"

cd "$PROJECT_ROOT"

# ─────────────────────────────────────────────────────────────────────
# Step 3: Playwright browser
# ─────────────────────────────────────────────────────────────────────

step "Step 3/6 — Installing Playwright Firefox"

# Install system deps for Playwright (Ubuntu)
info "Installing Playwright system dependencies..."
cd "$PROJECT_ROOT/mivzakim_scraper"
uv run playwright install-deps 2>/dev/null || apt-get install -y -qq \
    libxcb-shm0 libx11-xcb1 libx11-6 libxcb1 libxext6 libxrandr2 \
    libxcomposite1 libxcursor1 libxdamage1 libxfixes3 libxi6 \
    libgtk-3-0 libpangocairo-1.0-0 libpango-1.0-0 libatk1.0-0 \
    libcairo-gobject2 libcairo2 libgdk-pixbuf-2.0-0 libxrender1 \
    libasound2 libfreetype6 libfontconfig1 \
    2>/dev/null || info "Some Playwright deps may already be installed"

cd "$PROJECT_ROOT/mivzakim_scraper"
uv run playwright install firefox
ok "Firefox installed for Playwright"

cd "$PROJECT_ROOT"

# ─────────────────────────────────────────────────────────────────────
# Step 4: Set up PostgreSQL
# ─────────────────────────────────────────────────────────────────────

step "Step 4/6 — Setting up PostgreSQL"

# Create .env from template if it doesn't exist
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    ok "Created .env from .env.example"
else
    ok ".env already exists"
fi

# Decide mode
PG_MODE="skip"
if $SKIP_DB; then
    info "Skipped (--skip-db flag)"
elif $DRY_RUN; then
    if $DOCKER_AVAILABLE; then
        info "[DRY RUN] Would start PostgreSQL via Docker"
    else
        info "[DRY RUN] Would install and configure PostgreSQL natively"
    fi
elif $DOCKER_AVAILABLE; then
    PG_MODE="docker"
else
    PG_MODE="native"
fi

if [ "$PG_MODE" = "docker" ]; then
    # ── Docker mode ──────────────────────────────────────────────
    info "Using Docker mode"
    docker compose up -d
    ok "PostgreSQL container started"

    info "Waiting for PostgreSQL to be ready..."
    RETRIES=30
    until docker exec sentisense-postgres pg_isready -U "$DB_USER" -d "$DB_NAME" &>/dev/null || [ $RETRIES -eq 0 ]; do
        sleep 1
        RETRIES=$((RETRIES - 1))
    done

    if [ $RETRIES -eq 0 ]; then
        fail "PostgreSQL did not become ready in time"
        docker compose logs postgres
        exit 1
    fi
    ok "PostgreSQL is healthy and ready"

elif [ "$PG_MODE" = "native" ]; then
    # ── Native mode (no Docker — e.g. inside a container) ───────
    info "Using native PostgreSQL mode (no Docker daemon detected)"

    # Install PostgreSQL if not present
    if ! check_cmd psql; then
        info "Installing PostgreSQL..."
        apt-get update -qq
        apt-get install -y -qq postgresql postgresql-client
        ok "PostgreSQL installed"
    fi

    # Detect installed version
    PG_VERSION=$(ls /etc/postgresql/ 2>/dev/null | sort -V | tail -1)
    if [ -z "$PG_VERSION" ]; then
        fail "PostgreSQL installed but no cluster found"
        exit 1
    fi

    # Start PostgreSQL (no systemd in containers — use pg_ctlcluster)
    if pg_isready -q 2>/dev/null; then
        ok "PostgreSQL $PG_VERSION already running"
    else
        info "Starting PostgreSQL $PG_VERSION..."
        pg_ctlcluster "$PG_VERSION" main start 2>/dev/null || true
        RETRIES=15
        until pg_isready -q 2>/dev/null || [ $RETRIES -eq 0 ]; do
            sleep 1
            RETRIES=$((RETRIES - 1))
        done
        if [ $RETRIES -eq 0 ]; then
            fail "PostgreSQL did not start in time"
            exit 1
        fi
        ok "PostgreSQL $PG_VERSION started"
    fi

    # Enable TCP listening (psycopg connects via localhost)
    PG_CONF="/etc/postgresql/$PG_VERSION/main/postgresql.conf"
    PG_HBA="/etc/postgresql/$PG_VERSION/main/pg_hba.conf"

    if grep -q "^listen_addresses" "$PG_CONF" 2>/dev/null; then
        ok "TCP listening already configured"
    else
        info "Enabling TCP listening on localhost..."
        sed -i "s/#listen_addresses = 'localhost'/listen_addresses = 'localhost'/" "$PG_CONF"
        ok "listen_addresses = 'localhost'"
    fi

    # Ensure md5 auth for local TCP connections
    if grep -q "host.*all.*all.*127.0.0.1.*md5\|host.*all.*all.*127.0.0.1.*scram" "$PG_HBA" 2>/dev/null; then
        ok "TCP authentication already configured"
    else
        info "Configuring md5 authentication for localhost..."
        echo "host all all 127.0.0.1/32 md5" >> "$PG_HBA"
        echo "host all all ::1/128 md5" >> "$PG_HBA"
        ok "md5 auth enabled"
    fi

    # Restart to apply config
    pg_ctlcluster "$PG_VERSION" main restart 2>/dev/null || true
    sleep 1

    # Create role (idempotent)
    if su - postgres -c "psql -tc \"SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'\"" | grep -q 1; then
        ok "Role '$DB_USER' already exists"
    else
        su - postgres -c "psql -c \"CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';\""
        ok "Role '$DB_USER' created"
    fi

    # Create database (idempotent)
    if su - postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname='$DB_NAME'\"" | grep -q 1; then
        ok "Database '$DB_NAME' already exists"
    else
        su - postgres -c "psql -c \"CREATE DATABASE $DB_NAME OWNER $DB_USER;\""
        ok "Database '$DB_NAME' created"
    fi

    # Load schema
    info "Loading schema from init_db.sql..."
    su - postgres -c "psql -d $DB_NAME < $PROJECT_ROOT/scripts/init_db.sql" 2>/dev/null || true

    # Grant permissions (tables owned by postgres superuser need explicit grants)
    su - postgres -c "psql -d $DB_NAME -c 'GRANT ALL ON ALL TABLES IN SCHEMA public TO $DB_USER; GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO $DB_USER;'" 2>/dev/null
    ok "Permissions granted to '$DB_USER'"
fi

# Verify schema (both modes)
if [ "$PG_MODE" != "skip" ]; then
    info "Verifying schema..."
    run_psql -c "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;" 2>/dev/null | head -10
    ok "Schema verified"
fi

# ─────────────────────────────────────────────────────────────────────
# Step 5: Update data.csv (scrape new dates)
# ─────────────────────────────────────────────────────────────────────

step "Step 5/6 — Updating data.csv"

if $SKIP_SCRAPE; then
    info "Skipped (--skip-scrape flag)"
elif $DRY_RUN; then
    cd "$PROJECT_ROOT/processing_engine"
    uv run python "$PROJECT_ROOT/scripts/update_data_csv.py" --dry-run
    cd "$PROJECT_ROOT"
else
    if [ -f "$PROJECT_ROOT/data.csv" ]; then
        BEFORE_ROWS=$(wc -l < "$PROJECT_ROOT/data.csv")
        info "Current data.csv: $BEFORE_ROWS lines"
    else
        info "No data.csv found — will create fresh"
    fi

    cd "$PROJECT_ROOT/processing_engine"
    uv run python "$PROJECT_ROOT/scripts/update_data_csv.py"
    cd "$PROJECT_ROOT"

    if [ -f "$PROJECT_ROOT/data.csv" ]; then
        AFTER_ROWS=$(wc -l < "$PROJECT_ROOT/data.csv")
        ok "data.csv updated: $AFTER_ROWS lines"
    fi
fi

# ─────────────────────────────────────────────────────────────────────
# Step 6: Migrate CSV → PostgreSQL
# ─────────────────────────────────────────────────────────────────────

step "Step 6/6 — Migrating data.csv → PostgreSQL"

if $SKIP_MIGRATE; then
    info "Skipped (--skip-migrate flag)"
elif $DRY_RUN; then
    cd "$PROJECT_ROOT/processing_engine"
    uv run python "$PROJECT_ROOT/scripts/migrate_csv_to_db.py" --dry-run
    cd "$PROJECT_ROOT"
elif [ "$PG_MODE" = "skip" ]; then
    info "Skipped (PostgreSQL was not set up in Step 4)"
elif [ ! -f "$PROJECT_ROOT/data.csv" ]; then
    fail "data.csv not found — cannot migrate. Run the scraper first."
else
    cd "$PROJECT_ROOT/processing_engine"
    uv run python "$PROJECT_ROOT/scripts/migrate_csv_to_db.py"
    cd "$PROJECT_ROOT"

    # Show final stats
    echo ""
    info "Database stats:"
    run_psql -c "
        SELECT
            COUNT(*)                     AS total_rows,
            COUNT(DISTINCT date)         AS unique_dates,
            MIN(date)::text              AS earliest_date,
            MAX(date)::text              AS latest_date
        FROM raw_headlines;
    " 2>/dev/null
fi

# ─────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✓ SentiSense pipeline initialization complete!${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
echo ""
echo "  PostgreSQL:  localhost:5432  (user: $DB_USER / db: $DB_NAME)"
echo "  Logs:        $PROJECT_ROOT/logs/"
echo ""
echo "  Next steps:"
echo "    • Run evaluations:  cd processing_engine && uv run python -m evaluation.evaluate --all-models"
echo "    • Daily cron:       cd processing_engine && uv run python ../scripts/daily_scrape_to_db.py"
echo ""
