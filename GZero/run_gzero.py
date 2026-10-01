"""G-Zero's loop, run: ``python GZero/run_gzero.py``.

First one run (seed 0), printing both agents after each phase of each round.
Then 10 seeds, changing one of the paper's ingredients at a time:

    G-Zero           as in the paper: r = delta - P_length - P_BLEU, the lower-50% filter, DPO
    no filter        DPO on every pair, not only the lower 50% of delta
    no P_BLEU        no penalty for picking the same (query, hint) as the rest of the batch

Takes about half a minute. ``RL_IMPL=scratch`` runs your from_scratch code.
"""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))     # your implementation
sys.path.insert(1 if os.getenv("RL_IMPL") == "scratch" else 0, str(HERE))
import gzero as impl  # noqa: E402
import gzero_env as env  # noqa: E402

SEEDS = 10
HINT_NAME = ["position 0", "position 1", "position 2", "all"]


def show(rnd, record):
    print(f"\nround {rnd}")
    print("  Phase 1  Proposer pi_P, p(query 0..5): "
          + "  ".join(f"q{q} {p:.2f}" for q, p in enumerate(record["by_query"]))
          + "   hint kind: " + "  ".join(f"{name} {p:.2f}" for name, p in zip(HINT_NAME, record["by_hint"])))
    print(f"  Phase 2  {record['pairs']} DPO pairs (the lower half of 200 by delta); Generator pi_G, unassisted p(good):")
    for q in range(env.QUERIES):
        before = " ".join(f"{float(p):.2f}" for p in record["p_good_before"][q])
        after = " ".join(f"{float(p):.2f}" for p in record["p_good"][q])
        print(f"           query {q} (blind spots {str(env.BLIND[q]):<9}) {before}  ->  {after}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    start = env.Generator().p_good()
    print("G-Zero on the toy: 6 queries of 3 tokens, a Generator of 6 x 3 x 4 logits, a Proposer of 24 logits.")
    print(f"Before training the Generator's unassisted p(good) averages {float(start.mean()):.3f}; "
          "its blind spots start at 0.13, the rest at 0.87.")
    print("Each round: Phase 1 (30 GRPO steps on pi_P), Phase 2 (200 pairs, lower half kept, 50 DPO steps on pi_G).")
    _, _, history = impl.gzero(seed=0, report=show)
    print(f"\nOverall unassisted p(good): {float(start.mean()):.3f} -> "
          + " -> ".join(f"{float(r['p_good'].mean()):.3f}" for r in history))

    print(f"\n{SEEDS} seeds, 2 rounds each:\n")
    print(f"  {'run':<10} | {'p(good) after 1, 2':<18} | {'pi_P on query 5':<15} | {'positions worse':<15} | top output")
    rows = {"G-Zero": {}, "no filter": {"use_filter": False}, "no P_BLEU": {"use_bleu": False}}
    for name, kwargs in rows.items():
        runs = [impl.gzero(seed=seed, **kwargs)[2] for seed in range(SEEDS)]
        quality = [sum(float(h[i]["p_good"].mean()) for h in runs) / SEEDS for i in range(2)]
        query5 = [sum(h[i]["by_query"][5] for h in runs) / SEEDS for i in range(2)]
        worse = sum(int(((start - h[-1]["p_good"]) > 0.1).sum()) for h in runs) / SEEDS
        top = sum(h[-1]["top_output"] for h in runs) / SEEDS
        print(f"  {name:<10} | {quality[0]:.3f} {quality[1]:.3f}        | {query5[0]:.2f} -> {query5[1]:.2f}    | "
              f"{worse:>6.1f} of 18    | {top:.2f}", flush=True)
    print("""
  pi_P on query 5: the Proposer's probability on the query with three blind spots, after each round.
  positions worse: of the 18 (query, position) cells, how many ended more than 0.1 below their start.
  top output: pi_P's largest probability on any one of its 24 (query, hint) outputs (1/24 = 0.04 is uniform).

Reading it:
  G-Zero     the Generator improves with no answer ever checked. The Proposer first aims at
             query 5, the most blind spots, then moves off it as DPO fixes it.
  no filter  better on this toy, and fewer cells slip back. The lower-50% filter drops the
             high-delta pairs, which here are the ones with the most to teach; its purpose,
             keeping DPO inside its KL budget on an LLM, has nothing to protect in a table.
  no P_BLEU  the Proposer piles onto one (query, hint) and the Generator barely improves:
             each query's blind spots are separate, so variety is what spreads the lessons.""")
