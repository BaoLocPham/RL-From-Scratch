"""Agent0's two agents, end to end: ``python Agent0/steps_agent0.py``.

Agent0 trains two models against each other with two different algorithms, and
each half has its own walkthrough:

    Agent0/steps_curriculum.py   the proposer -- writes questions, plain GRPO
    Agent0/steps_executor.py     the solver   -- answers them, ADPO

This file runs both in order, which is also the order they depend on: the
Curriculum Agent's Step 4 pass produces the difficulty label that ADPO consumes.
Run either file on its own if you only want one half.

``RL_IMPL=scratch`` applies to the curriculum half, whose six functions are the
``from_scratch`` exercise. ADPO is reference material and never switches.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

for half in ("steps_curriculum.py", "steps_executor.py"):
    result = subprocess.run([sys.executable, str(HERE / half)])
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print()
