#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x camera_experiment/.venv/bin/python ]; then
  uv venv camera_experiment/.venv
  uv --directory camera_experiment pip install --python .venv/bin/python -r requirements.txt
fi

credentials="${XDG_CONFIG_HOME:-$HOME/.config}/captchasolver/credentials.env"
if { [ -z "${CLOUDFLARE_ACCOUNT_ID:-}" ] || [ -z "${CLOUDFLARE_API_TOKEN:-}" ]; } && [ -r "$credentials" ]; then
  source "$credentials"
fi

export PYTHONDONTWRITEBYTECODE=1
exec camera_experiment/.venv/bin/python -m camera_experiment.gallery "$@"
