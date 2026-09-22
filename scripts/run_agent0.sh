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
# Agent0 only, since it alone trains two agents with two algorithms:
#
#   overview    the iteration flow diagram, and the glossary of terms
#   trace       ONE question followed through Step 3 and Step 4, narrated
#   curriculum  the proposer's half -- writes questions, plain GRPO
#   executor    the solver's half   -- answers them, ADPO
#
# For Agent0, 'steps' runs both of those in dependency order.
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
    sed -n '3,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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

cmd_half() {
    if [ "$MODULE" != "Agent0" ]; then
        echo "'$1' is Agent0 only; $MODULE has a single agent" >&2
        return 1
    fi
    banner "Agent0: the $1 half"
    "$PYTHON" "Agent0/steps_$1.py"
}

cmd_overview() {
    if [ "$MODULE" != "Agent0" ]; then
        echo "'overview' is Agent0 only" >&2
        return 1
    fi
    banner "Agent0: the iteration flow, and the terms"
    "$PYTHON" "Agent0/overview.py"
}

cmd_trace() {
    if [ "$MODULE" != "Agent0" ]; then
        echo "'trace' is Agent0 only" >&2
        return 1
    fi
    banner "Agent0: one question through Step 3 and Step 4"
    "$PYTHON" "Agent0/trace_one_question.py"
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
    run)        cmd_run ;;
    curriculum) cmd_half curriculum ;;
    executor)   cmd_half executor ;;
    overview)   cmd_overview ;;
    trace)      cmd_trace ;;
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
