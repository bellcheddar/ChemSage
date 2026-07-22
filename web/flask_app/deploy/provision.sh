#!/usr/bin/env bash
# One-time droplet provisioning for ChemSage web app. Run as root ON the droplet after
# pushing code with deploy.sh:
#
#   sudo SERVER_NAME=chemsage.mdeller.com bash /opt/chemsage/deploy/provision.sh
#
# Idempotent: safe to re-run.
set -euo pipefail

APP_DIR=/opt/chemsage
APP_USER=chemsage
BIND_ADDR="127.0.0.1:8001"
SERVER_NAME="${SERVER_NAME:-chemsage.mdeller.com}"

echo "==> ChemSage provisioning for ${SERVER_NAME}"

if [[ $EUID -ne 0 ]]; then echo "Run as root."; exit 1; fi
if [[ ! -f "$APP_DIR/app.py" ]]; then
  echo "No code at $APP_DIR — push it first: bash deploy/deploy.sh"; exit 1
fi

echo "==> Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip python3-dev build-essential \
  nginx certbot python3-certbot-nginx rsync

echo "==> Creating service user '${APP_USER}'"
id -u "$APP_USER" &>/dev/null || useradd --system --home "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

echo "==> Building Python venv"
if [[ ! -x "$APP_DIR/.venv/bin/python" ]]; then
  sudo -u "$APP_USER" python3 -m venv "$APP_DIR/.venv"
fi
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"

echo "==> Creating .env (HF Space URL)"
if [[ ! -f "$APP_DIR/.env" ]]; then
  cat > "$APP_DIR/.env" <<'EOF'
HF_SPACE_URL=https://dellboy-chem-sage-api.hf.space
SECRET_KEY=change-me
EOF
  chown "$APP_USER:$APP_USER" "$APP_DIR/.env"
  chmod 600 "$APP_DIR/.env"
  echo "    Created $APP_DIR/.env — edit SECRET_KEY before going live."
fi

echo "==> Installing systemd unit"
cp "$APP_DIR/deploy/chemsage-web.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now chemsage-web.service

echo "==> Installing nginx site"
sed -e "s|__SERVER_NAME__|${SERVER_NAME}|g" \
  "$APP_DIR/deploy/nginx-chemsage.conf" > /etc/nginx/sites-available/chemsage
ln -sf /etc/nginx/sites-available/chemsage /etc/nginx/sites-enabled/chemsage
nginx -t && systemctl reload nginx

echo "==> Requesting TLS certificate"
if certbot certificates 2>/dev/null | grep -q "$SERVER_NAME"; then
  echo "    Certificate for ${SERVER_NAME} already present; skipping."
else
  certbot --nginx -d "$SERVER_NAME" --non-interactive --agree-tos \
    -m "marc@marcdeller.com" --redirect || \
    echo "    certbot failed (DNS not pointed yet?). Re-run: certbot --nginx -d ${SERVER_NAME}"
fi

echo "==> Done."
systemctl --no-pager --lines=5 status chemsage-web.service || true
