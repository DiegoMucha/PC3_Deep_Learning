#!/usr/bin/env bash
# Usar: bash data/download_data.sh [directorio de destino]
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python -m src.utils download --destination "${1:-data/raw}"
