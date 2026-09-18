#!/usr/bin/env bash
#
# Run one module.  ./scripts/run_agent0.sh <command>
#
# Sibling links scripts/run_ppo.sh and scripts/run_grpo.sh drive PPO and GRPO;
# the module is taken from this script's own filename.
#
#   check     grade the module's from_scratch exercise, stopping at the first gap
#   steps     the walkthrough, against the reference implementation
#   scratch   the walkthrough, against your from_scratch implementation
#   run       the runnable demonstration
#   diff      prove your implementation matches: check, then steps vs scratch
#   all       check, steps, run
#
# With no command, prints this list.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# run_agent0.sh -> Agent0, run_ppo.sh -> PPO, run_grpo.sh -> GRPO
STEM="$(basename "${BASH_SOURCE[0]}" .sh)"; STEM="${STEM#run_}"
case "$STEM" in
    agent0) MODULE=Agent0 ;;
    ppo)    MODULE=PPO ;;
    grpo)   MODULE=GRPO ;;
    *)      echo "unknown module: $STEM" >&2; exit 1 ;;
esac

PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
if [ -z "$PYTHON" ]; then
    echo "no python on PATH; set PYTHON=/path/to/python" >&2
    exit 1
fi

usage() {
    sed -n '3,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

banner() {
    printf '\n\033[1m== %s\033[0m\n' "$1"
}

cmd_check() {
    banner "grading $MODULE/from_scratch/$STEM.py"
    "$PYTHON" "$MODULE/from_scratch/check.py"
}

cmd_steps() {
    banner "walkthrough (reference implementation)"
    "$PYTHON" "$MODULE/steps_$STEM.py"
}

cmd_scratch() {
    banner "walkthrough (your implementation)"
    RL_IMPL=scratch "$PYTHON" "$MODULE/steps_$STEM.py"
}

cmd_run() {
    banner "$MODULE demo"
    "$PYTHON" "$MODULE/run_$STEM.py"
}

cmd_diff() {
    cmd_check
    banner "reference output vs yours"
    local reference mine
    reference="$(mktemp)"
    mine="$(mktemp)"
    trap 'rm -f "$reference" "$mine"' RETURN
    "$PYTHON" "$MODULE/steps_$STEM.py" > "$reference"
    RL_IMPL=scratch "$PYTHON" "$MODULE/steps_$STEM.py" > "$mine"
    if diff -u "$reference" "$mine"; then
        echo "identical -- your implementation is indistinguishable from common.py"
    else
        echo "the two differ; the lines above are reference vs yours" >&2
        return 1
    fi
}

cmd_all() {
    cmd_check
    cmd_steps
    cmd_run
}

case "${1:-}" in
    check)   cmd_check   ;;
    steps)   cmd_steps   ;;
    scratch) cmd_scratch ;;
    run)     cmd_run     ;;
    diff)    cmd_diff    ;;
    all)     cmd_all     ;;
    ""|-h|--help|help) usage ;;
    *)
        echo "unknown command: $1" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac
