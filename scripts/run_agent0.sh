#!/usr/bin/env bash
#
# Run the Agent0 module.  ./scripts/run_agent0.sh <command>
#
#   check     grade from_scratch/agent0.py, stopping at the first unfinished stage
#   steps     the walkthrough, against the reference implementation
#   scratch   the walkthrough, against your from_scratch implementation
#   run       one full iteration: Steps 3, 4 and 5 end to end
#   diff      prove your implementation matches: check, then steps vs scratch
#   all       check, steps, run
#
# With no command, prints this list.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
if [ -z "$PYTHON" ]; then
    echo "no python on PATH; set PYTHON=/path/to/python" >&2
    exit 1
fi

usage() {
    sed -n '3,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

banner() {
    printf '\n\033[1m== %s\033[0m\n' "$1"
}

cmd_check() {
    banner "grading Agent0/from_scratch/agent0.py"
    "$PYTHON" Agent0/from_scratch/check.py
}

cmd_steps() {
    banner "walkthrough (reference implementation)"
    "$PYTHON" Agent0/steps_agent0.py
}

cmd_scratch() {
    banner "walkthrough (your implementation)"
    RL_IMPL=scratch "$PYTHON" Agent0/steps_agent0.py
}

cmd_run() {
    banner "one Agent0 iteration: Steps 3, 4 and 5"
    "$PYTHON" Agent0/run_agent0.py
}

cmd_diff() {
    cmd_check
    banner "reference output vs yours"
    local reference mine
    reference="$(mktemp)"
    mine="$(mktemp)"
    trap 'rm -f "$reference" "$mine"' RETURN
    "$PYTHON" Agent0/steps_agent0.py > "$reference"
    RL_IMPL=scratch "$PYTHON" Agent0/steps_agent0.py > "$mine"
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
