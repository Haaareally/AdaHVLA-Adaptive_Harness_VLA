#!/usr/bin/env bash
# Isaac Lab fork required by NaVILA: https://github.com/yang-zj1026/IsaacLab
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: ISAACLAB_ROOT=/path/to/IsaacLab bash scripts/setup_locomotion.sh

Use an activated Python 3.10 environment and an existing NaVILA Isaac Lab 1.1.0
checkout. This installs Go2 simulation dependencies and AdaHVLA; no VLA models.
ADAHVLA_CUDA selects cu121 (default) or cu118 for the PyTorch wheels.
EOF
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo "Unknown arguments; use --help." >&2
    exit 2
fi

: "${ISAACLAB_ROOT:?Set ISAACLAB_ROOT to the existing NaVILA Isaac Lab 1.1.0 checkout}"
ISAACLAB_ROOT="$(cd -- "$ISAACLAB_ROOT" && pwd)"
ADAHVLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ADAHVLA_CUDA="${ADAHVLA_CUDA:-cu121}"
case "$ADAHVLA_CUDA" in
    cu118|cu121) ;;
    *) echo "ADAHVLA_CUDA must be cu118 or cu121." >&2; exit 2 ;;
esac

python - "$ISAACLAB_ROOT" <<'PY'
import sys
from pathlib import Path

if sys.version_info[:2] != (3, 10):
    raise SystemExit("Go2 simulation requires an activated Python 3.10 environment.")
root = Path(sys.argv[1])
version = root / "VERSION"
package = root / "source/extensions/omni.isaac.lab"
if not version.is_file() or version.read_text().strip() != "1.1.0":
    raise SystemExit("ISAACLAB_ROOT must contain Isaac Lab VERSION 1.1.0.")
if not (package / "omni/isaac/lab/app/app_launcher.py").is_file():
    raise SystemExit("ISAACLAB_ROOT does not provide the required omni.isaac.lab API.")
if not (package / "pyproject.toml").is_file():
    raise SystemExit("The Lab checkout must include pyproject.toml with its build dependencies.")
if not (root / "source/apps/isaaclab.python.headless.rendering.kit").is_file():
    raise SystemExit("ISAACLAB_ROOT must include the Lab application configuration files.")
PY

cd -- "$ADAHVLA_ROOT"
python -m pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url "https://download.pytorch.org/whl/$ADAHVLA_CUDA"
# Install only Lab core: its global installer also discovers unrelated extensions.
# Lab declares setuptools, wheel and toml in pyproject.toml; keep build isolation.
python -m pip install -r requirements-locomotion.txt \
    -e "$ISAACLAB_ROOT/source/extensions/omni.isaac.lab"
python -m pip check
