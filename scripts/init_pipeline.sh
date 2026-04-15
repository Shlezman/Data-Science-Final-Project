#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# SentiSense — Full Pipeline Initialization Script (Ubuntu)
# ═══════════════════════════════════════════════════════════════════════
#
# This script bootstraps the entire SentiSense data pipeline from zero:
#   1. Installs system dependencies (Docker, Python, uv)
#   2. Installs Python packages via uv for all modules
#   3. Installs Playwright browser for the scraper
#   4. Starts PostgreSQL via docker compose
#   5. Updates data.csv with newly scraped headlines
#   6. Migrates the full dataset into PostgreSQL
#
# Usage:
#   chmod +x scripts/init_pipeline.sh
#   ./scripts/init_pipeline.sh
#
# Flags:
#   --skip-scrape    Skip the data.csv update (scraper) step
#   --skip-migrate   Skip the CSV-to-DB migration step
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
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --skip-scrape)  SKIP_SCRAPE=true ;;
        --skip-migrate) SKIP_MIGRATE=true ;;
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

cd "$PROJECT_ROOT"

# ─────────────────────────────────────────────────────────────────────
# Step 1: System dependencies
# ─────────────────────────────────────────────────────────────────────

step "Step 1/6 — Checking system dependencies"

# Docker
if check_cmd docker; then
    docker --version
else
    info "Installing Docker..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker.io docker-compose-plugin
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER"
    ok "Docker installed. NOTE: You may need to log out and back in for group changes."
fi

# Docker Compose (v2 plugin)
if docker compose version &>/dev/null; then
    ok "docker compose v2: $(docker compose version --short)"
else
    info "Installing docker-compose-plugin..."
    sudo apt-get install -y -qq docker-compose-plugin
    ok "docker-compose-plugin installed"
fi

# Python 3.12+
if check_cmd python3; then
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if python3 -c "import sys; exit(0 if sys.version_info >= (3,12) else 1)"; then
        ok "Python $PY_VERSION (>= 3.12)"
    else
        fail "Python $PY_VERSION found but 3.12+ required"
        info "Install with: sudo apt install python3.12 python3.12-venv"
        exit 1
    fi
else
    fail "Python3 not found"
    info "Install with: sudo apt install python3.12 python3.12-venv"
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
sudo npx playwright install-deps firefox 2>/dev/null || sudo apt-get install -y -qq \
    libgtk-3-0 libnotify4 libnss3 libxss1 libasound2t64 libatk-bridge2.0-0 libdrm2 \
    libgbm1 libpango-1.0-0 libcairo2 libcups2 libx11-xcb1 libxcomposite1 \
    libxdamage1 libxrandr2 2>/dev/null || info "Some Playwright deps may already be installed"

cd "$PROJECT_ROOT/mivzakim_scraper"
uv run playwright install firefox
ok "Firefox installed for Playwright"

cd "$PROJECT_ROOT"

# ─────────────────────────────────────────────────────────────────────
# Step 4: Start PostgreSQL
# ─────────────────────────────────────────────────────────────────────

step "Step 4/6 — Starting PostgreSQL"

# Create .env from template if it doesn't exist
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    ok "Created .env from .env.example"
else
    ok ".env already exists"
fi

if $DRY_RUN; then
    info "[DRY RUN] Would run: docker compose up -d"
else
    docker compose up -d
    ok "PostgreSQL container started"

    # Wait for healthy
    info "Waiting for PostgreSQL to be ready..."
    RETRIES=30
    until docker exec sentisense-postgres pg_isready -U sentisense -d sentisense &>/dev/null || [ $RETRIES -eq 0 ]; do
        sleep 1
        RETRIES=$((RETRIES - 1))
    done

    if [ $RETRIES -eq 0 ]; then
        fail "PostgreSQL did not become ready in time"
        docker compose logs postgres
        exit 1
    fi

    ok "PostgreSQL is healthy and ready"

    # Show tables created by init_db.sql
    info "Verifying schema..."
    docker exec sentisense-postgres psql -U sentisense -d sentisense -c \
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;" \
        2>/dev/null | head -10
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
elif [ ! -f "$PROJECT_ROOT/data.csv" ]; then
    fail "data.csv not found — cannot migrate. Run the scraper first."
else
    cd "$PROJECT_ROOT/processing_engine"
    uv run python "$PROJECT_ROOT/scripts/migrate_csv_to_db.py"
    cd "$PROJECT_ROOT"

    # Show final stats
    echo ""
    info "Database stats:"
    docker exec sentisense-postgres psql -U sentisense -d sentisense -c "
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
echo "  PostgreSQL:  localhost:5432  (user: sentisense / db: sentisense)"
echo "  Logs:        $PROJECT_ROOT/logs/"
echo ""
echo "  Next steps:"
echo "    • Run evaluations:  cd processing_engine && uv run python -m evaluation.evaluate --all-models"
echo "    • Daily cron:       cd processing_engine && uv run python scripts/daily_scrape_to_db.py"
echo "    • pgAdmin UI:       docker compose --profile admin up -d  →  http://localhost:5050"
echo ""
