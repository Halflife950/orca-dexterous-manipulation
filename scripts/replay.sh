#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
axis="${1:?Usage: replay.sh x|y|z OUTPUT_DIRECTORY}"
dest="${2:?Specify a scratch directory}"
mkdir -p "$dest"
dest="$(cd "$dest" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$root/methods/orca_reversal/simulator${PYTHONPATH:+:$PYTHONPATH}"
audit="$root/methods/experiments/forearm_ppo_pair/support_audit"
if [[ "$axis" == y || "$axis" == z ]]; then
    export ORCA_GAP_LIBRARY="$dest/gap.so"
    g++ -std=c++14 -O3 -shared -fPIC -I/usr/include/eigen3 "$audit/gap.cpp" -lfcl -lccd -o "$ORCA_GAP_LIBRARY"
fi
case "$axis" in
  x)
    for bank in 731407 731411 731417; do
      python "$root/methods/experiments/ppo_factorial/evaluate.py" "$root/policies/x.pth" tilt "$bank" "$dest" --device "${ORCA_DEVICE:-cuda:0}"
    done ;;
  y)
    export ORCA_EVAL_OUTPUT="$dest"
    python "$root/methods/experiments/forearm_ppo_pair/study.py" test Y free 976043
    python "$audit/capture.py" "$root/methods" Y free 976043 "$dest/contacts"
    ORCA_AUDIT_SEED=976043 ORCA_AUDIT_AXIS=Y ORCA_AUDIT_MODE=free "${ORCA_SCORE_PYTHON:-/usr/bin/python3}" "$audit/score.py" "$root/methods" "$dest/contacts" "$dest/scores" ;;
  z)
    program="$root/methods/experiments/inhand_z_spin/outward_slip_ablation/experiment.py"
    python "$program" test 990037 "$dest" --fresh
    ORCA_SCORE_OUTPUT="$dest/scores" "${ORCA_SCORE_PYTHON:-/usr/bin/python3}" "$program" score 990037 "$dest" --fresh ;;
  *) echo 'Choose x, y or z.' >&2; exit 2 ;;
esac
