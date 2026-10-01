"""DPO on the multi-step toy, and what each of its choices does: ``python SimpleDPO/run_simple_dpo.py``.

Seven runs of the same budget, 20 seeds each, 60 iterations of 8 questions x 2 attempts
(16 attempts, SimpleGRPO's budget):

    DPO                pairs sampled from pi_ref before training, the better attempt chosen, beta 0.1, 10 epochs
    1 epoch            each batch of pairs used once
    online             each iteration's pairs sampled from the policy as it is now (verl's online-DPO recipe)
    by outcome         pairs from pi_ref, labelled by the rewards they actually got (+1 / -1, -0.1 per search)
    by outcome, online the same, sampled from the policy
    GRPO G 2, Dr.GRPO  SimpleGRPO on the same 16 attempts per iteration, (R - pair mean)
    GRPO G 2           the same, divided by the pair's std: +0.71 or -0.71

Takes about two minutes. ``RL_IMPL=scratch`` runs your from_scratch code.
"""

import importlib.util
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))     # your implementation
sys.path.insert(1 if os.getenv("RL_IMPL") == "scratch" else 0, str(HERE))
import pair_env as env  # noqa: E402
import simple_dpo as impl  # noqa: E402
import group_env  # noqa: E402  (SimpleGRPO's toy; pair_env put it on the path)

# The GRPO rows always run the reference SimpleGRPO, loaded under its own name.
_spec = importlib.util.spec_from_file_location("simple_grpo_reference", HERE.parent / "SimpleGRPO" / "simple_grpo.py")
grpo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grpo)

SEEDS = 20
MARKS = (9, 29, 59)                                    # after 10, 30 and 60 iterations


def train_grpo(scale_by_std):
    def run(seed):
        advantage = lambda batch: grpo.group_advantage(batch["rewards"], 2, scale_by_std=scale_by_std)  # noqa: E731
        curve, policy, _ = group_env.train(grpo, seed, questions=8, group_size=2, advantage=advantage)
        return curve, policy
    return run


RUNS = {
    "DPO": (lambda seed: env.train(impl, seed), "from pi_ref, better wins"),
    "1 epoch": (lambda seed: env.train(impl, seed, epochs=1), "each batch used once"),
    "online": (lambda seed: env.train(impl, seed, online=True), "pairs from the policy"),
    "by outcome": (lambda seed: env.train(impl, seed, labels="outcome"), "labelled by reward got"),
    "outcome, onl.": (lambda seed: env.train(impl, seed, labels="outcome", online=True), "the same, online"),
    "GRPO G 2, Dr": (train_grpo(False), "R - pair mean"),
    "GRPO G 2": (train_grpo(True), "(R - mean) / std"),
}


def pairs_kept(labels, seeds=SEEDS, iterations=60, questions=8):
    """How many of the 480 questions give a pair: the rest are two equally good attempts, skipped."""
    total = 0
    for seed in range(seeds):
        torch.manual_seed(seed)
        ref = env.Policy()
        total += sum(env.collect_pairs(ref, ref, questions, impl.sequence_logp, labels)["chosen_states"].shape[0]
                     for _ in range(iterations))
    return total / seeds


if __name__ == "__main__":
    torch.set_num_threads(1)                           # tiny model: one thread is fastest
    kept, kept_outcome = pairs_kept("better"), pairs_kept("outcome")
    print(f"""The multi-step toy, two attempts per question, labelled: which attempt is better.
HARD needs two searches, EASY none. Start J {env.true_reward(env.Policy()):.3f}, best possible {env.BEST_J}.
{SEEDS} seeds, 60 iterations of 8 questions x 2 attempts, SGD lr 0.3 / beta = 3.0, 10 epochs of minibatches of 4 pairs.
Of the 480 questions, about {kept:.0f} give a pair ({kept_outcome:.0f} when labelled by outcome); the rest are ties, skipped.
"stuck": J below 0.7 after 60 iterations.
""")
    print(f"  {'run':<13} | {'what changes':<24} | {'J@10':>5} | {'J@30':>5} | {'J@60':>5} | stuck")
    results, policies = {}, {}
    for name, (run, what) in RUNS.items():
        runs = [run(seed) for seed in range(SEEDS)]
        mean = [sum(r[0][i] for r in runs) / SEEDS for i in MARKS]
        stuck = sum(r[0][-1] < 0.7 for r in runs)
        results[name], policies[name] = mean, [r[1] for r in runs]
        print(f"  {name:<13} | {what:<24} | " + " | ".join(f"{j:.3f}" for j in mean) + f" | {stuck:>2}/{SEEDS}",
              flush=True)

    print(f"""
Reading it:
  DPO          no reward, no critic, no rollouts while training: about {kept:.0f} labelled pairs from pi_ref,
               and a logistic regression on them. It reaches {results['DPO'][-1]:.3f} of a possible 0.900. A
               caveat: its labels come from each attempt's TRUE average, something the RL rows
               never get to see. That, more than DPO itself, is why it beats them here.
  1 epoch      DPO has no ratio to theta_old and no clip: its loss does not care who sampled the
               pairs, so reusing them needs no protection. One pass is simply less training
               ({results['1 epoch'][-1]:.3f}).
  online       pairs sampled from the policy as it improves, as verl's recipe does: about the
               same here ({results['online'][-1]:.3f}).
  by outcome   the fair comparison: labels from the rewards the attempts actually got, the same
               information GRPO sees. From pi_ref's pairs it still works ({results['by outcome'][-1]:.3f}): many
               are two searches against none, which the coin always decides the right way. Online
               it gets stuck ({results['outcome, onl.'][-1]:.3f}): once the policy searches once on HARD, every pair
               left is decided by a coin. One lucky search (0.9) beats two (0.8) half the time,
               and loses to none (-1.1 vs -1.0) half the time, so nothing pushes it on. A label
               says only WHICH attempt won, never by how much.
  GRPO G 2     the same failure, one module back: divided by the std, a pair's advantages are always
               +0.71 and -0.71, a sign and nothing more ({results['GRPO G 2'][-1]:.3f}). Without the divide,
               Dr.GRPO keeps the size of the gap ({results['GRPO G 2, Dr'][-1]:.3f}).""")

    learned = sum(env.implicit_rewards(impl, p, 0.1) for p in policies["DPO"]) / SEEDS
    print("\nHow DPO's policies score each number of searches, beta * log(pi / pi_ref), relative to 0, "
          "averaged over the 20 seeds:")
    print(f"  {'':<6} {'':<15} {'0':>6} {'1':>6} {'2':>6} {'3 searches':>10}")
    for qtype, label in ((env.HARD, "HARD"), (env.EASY, "EASY")):
        for i, (name, values) in enumerate((("average total", env.QUALITY[qtype] - env.QUALITY[qtype, 0]),
                                            ("DPO's score", learned[qtype]))):
            print(f"  {label if i == 0 else '':<6} {name:<15} " + " ".join(f"{float(v):>6.2f}" for v in values))
    print("  The same order as the average totals: trained only on which attempt was better, the policy has")
    print("  become a scorer of attempts. Its numbers are bigger, and keep growing, because the labels never")
    print("  disagree (walkthrough, step 7).")
    print("\nNext: DPO/ -- the same loss at verl's (batch, response_length) shapes, as its online-DPO recipe writes it.")
