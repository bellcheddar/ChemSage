#!/usr/bin/env bash
# Push ChemSage flask app from your Mac to the droplet and restart the service.
# Run from the web/flask_app directory (or the chem_sage repo root):
#
#   bash web/flask_app/deploy/deploy.sh
#
set -euo pipefail

DROPLET="${DROPLET_SSH:-root@45.55.102.228}"
APP_DIR=/opt/chemsage

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASK_APP="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$FLASK_APP/../.." && pwd)"

echo "==> Syncing code to ${DROPLET}:${APP_DIR}"
ssh "$DROPLET" "mkdir -p $APP_DIR"
rsync -az --delete \
  --exclude '.venv/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '.env' --exclude 'static/uploads/' \
  "$FLASK_APP/" "$DROPLET:$APP_DIR/"

# Also sync the repo's scripts/, rag/, and data/corpus/ dirs.
echo "==> Syncing chem_sage scripts and rag modules"
rsync -az --delete \
  --exclude '__pycache__/' --exclude '*.pyc' \
  "$REPO_ROOT/scripts/" "$DROPLET:$APP_DIR/../chem_sage_scripts/"
rsync -az --delete \
  --exclude '__pycache__/' --exclude '*.pyc' \
  "$REPO_ROOT/rag/" "$DROPLET:$APP_DIR/../chem_sage_rag/"
# Ensure /opt/rag → /opt/chem_sage_rag symlink exists (chat.py imports as "rag.*")
ssh "$DROPLET" "ln -sfn /opt/chem_sage_rag /opt/rag"

echo "==> Syncing corpus CSVs (corpus_lookup keyword fast-path)"
ssh "$DROPLET" "mkdir -p $APP_DIR/data/corpus"
rsync -az --delete \
  --exclude '__pycache__/' \
  "$REPO_ROOT/data/corpus/" "$DROPLET:$APP_DIR/data/corpus/"

echo "==> Installing dependencies + restarting service"
ssh "$DROPLET" bash -s <<REMOTE
set -euo pipefail
cd "${APP_DIR}"
if [[ ! -x .venv/bin/python ]]; then
  echo "No venv yet -- run deploy/provision.sh as root first."; exit 1
fi
sudo -u chemsage .venv/bin/pip install --quiet -r requirements.txt
find "${APP_DIR}" -not -path "${APP_DIR}/.venv/*" -exec chown chemsage:chemsage {} + 2>/dev/null || true
chown -R chemsage:chemsage "${APP_DIR}/../chem_sage_scripts/" "${APP_DIR}/../chem_sage_rag/" 2>/dev/null || true
chown -R chemsage:chemsage "${APP_DIR}/data/" 2>/dev/null || true
# rsync preserves Mac 0600 mode bits; ensure all scripts are readable by service user
chmod -R a+rX "${APP_DIR}/../chem_sage_scripts/" "${APP_DIR}/../chem_sage_rag/" 2>/dev/null || true
sudo systemctl restart chemsage-web.service
sudo systemctl --no-pager --lines=3 status chemsage-web.service || true
REMOTE

echo "==> Deployed to https://chemsage.mdeller.com"
