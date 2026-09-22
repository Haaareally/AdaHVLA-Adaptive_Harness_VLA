#!/usr/bin/env bash
set -euo pipefail

ADAHVLA_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Fill these placeholders, or export the same variables before running this file.
export ADAHVLA_API_KEY="${ADAHVLA_API_KEY:-YOUR_API_KEY}"
export ADAHVLA_BASE_URL="${ADAHVLA_BASE_URL:-https://YOUR_API_ENDPOINT/v1}"
export ADAHVLA_MODEL="${ADAHVLA_MODEL:-YOUR_VISION_MODEL}"
export ADAHVLA_VLA_HOST="${ADAHVLA_VLA_HOST:-127.0.0.1}"
export ADAHVLA_VLA_PORT="${ADAHVLA_VLA_PORT:-54321}"

export PYTHONDONTWRITEBYTECODE=1
exec "${ADAHVLA_PYTHON:-python}" "$ADAHVLA_PROJECT/scripts/run.py" "$@"
