# Sourced by every runner: repo root as working directory, output root in $QLENS_OUT (default ./outputs).
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export QLENS_OUT="${QLENS_OUT:-outputs}"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
OUT="$QLENS_OUT"
mkdir -p "$OUT"
