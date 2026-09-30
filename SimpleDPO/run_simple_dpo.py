"""DPO on the multi-step toy, and what each of its choices does: ``python SimpleDPO/run_simple_dpo.py``.

Ten runs of the same budget, 20 seeds each, 60 iterations of 8 pairs (16 attempts, SimpleGRPO's budget):

    DPO                pairs sampled from pi_ref before training, the rater's labels, beta 0.1, 10 epochs
    1 epoch            each batch of pairs used once
    beta 0.5           a stronger anchor to pi_ref (lr 0.6 = 0.3 / beta)
    beta 0.02          a weaker one (lr 15 = 0.3 / beta)
    online             each iteration's pairs sampled from the policy as it is now (verl's online-DPO recipe)
    online, beta 0.5   the same, anchored harder
    by outcome         pairs from pi_ref, ranked by the rewards they got (+1 / -1, -0.1 per search), ties dropped
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
TRUE = env.QUALITY - env.QUALITY[:, :1]                # the rater's r, shifted so that 0 searches is 0


def train_grpo(scale_by_std):
    def run(seed):
        advantage = lambda batch: grpo.group_advantage(batch["rewards"], 2, scale_by_std=scale_by_std)  # noqa: E731
        curve, policy, _ = group_env.train(grpo, seed, questions=8, group_size=2, advantage=advantage)
        return curve, policy
    return run


RUNS = {
    "DPO": (lambda seed: env.train(impl, seed), "pairs from pi_ref, rater", 0.1),
    "1 epoch": (lambda seed: env.train(impl, seed, epochs=1), "each batch used once", 0.1),
    "beta 0.5": (lambda seed: env.train(impl, seed, beta=0.5, lr=0.6), "anchored harder", 0.5),
    "beta 0.02": (lambda seed: env.train(impl, seed, beta=0.02, lr=15.0), "anchored less", 0.02),
    "online": (lambda seed: env.train(impl, seed, online=True), "pairs from the policy", 0.1),
    "online, b 0.5": (lambda seed: env.train(impl, seed, online=True, beta=0.5, lr=0.6), "the same, beta 0.5", 0.5),
    "by outcome": (lambda seed: env.train(impl, seed, labels="outcome"), "ranked by reward got", 0.1),
    "outcome, onl.": (lambda seed: env.train(impl, seed, labels="outcome", online=True), "the same, online", 0.1),
    "GRPO G 2, Dr": (train_grpo(False), "R - pair mean", None),
    "GRPO G 2": (train_grpo(True), "(R - mean) / std", None),
}

if __name__ == "__main__":
    torch.set_num_threads(1)                           # tiny model: one thread is fastest
    targets = {beta: env.true_reward(env.optimal_policy(beta)) for beta in (0.02, 0.1, 0.5)}
    print(f"""The multi-step toy, two attempts per question, and a rater who picks the better one.
HARD needs two searches, EASY none. Start J {env.true_reward(env.Policy()):.3f}, best possible {env.BEST_J}.
{SEEDS} seeds, 60 iterations of 8 pairs, SGD lr 0.3 / beta, 10 epochs of minibatches of 4 pairs.
"stuck": J below 0.7 after 60 iterations. "pi*_beta": the J of the policy DPO's loss is minimised by,
pi_ref * exp(r / beta). "r err": how far the implicit reward beta * log(pi / pi_ref) is from the rater's r,
averaged over both question types and 1-3 searches (0 = the reward recovered exactly).
""")
    print(f"  {'run':<13} | {'what changes':<24} | {'J@10':>5} | {'J@30':>5} | {'J@60':>5} | stuck | pi*_beta | r err")
    results, rewards, errors = {}, {}, {}
    for name, (run, what, beta) in RUNS.items():
        runs = [run(seed) for seed in range(SEEDS)]
        mean = [sum(r[0][i] for r in runs) / SEEDS for i in MARKS]
        stuck = sum(r[0][-1] < 0.7 for r in runs)
        target, error = "   -", "  -"
        if beta is not None:                           # the GRPO rows have no pi*_beta and no implicit reward
            learned = [env.implicit_rewards(impl, r[1], beta) for r in runs]
            rewards[name] = sum(learned) / SEEDS
            errors[name] = sum(float((r - TRUE)[:, 1:].abs().mean()) for r in learned) / SEEDS
            target, error = f"{targets[beta]:.3f}", f"{errors[name]:.2f}"
        results[name] = mean
        print(f"  {name:<13} | {what:<24} | " + " | ".join(f"{j:.3f}" for j in mean) +
              f" | {stuck:>2}/{SEEDS} | {target:>8} | {error:>5}", flush=True)

    print(f"""
Reading it:
  DPO          no reward, no critic, no rollouts while training: 480 judged pairs from pi_ref, and a
               logistic regression on them. It ends below its target ({results['DPO'][-1]:.3f} vs {targets[0.1]:.3f}) mostly
               because the rater barely separates close answers: two searches over three on HARD
               only 52% of the time, none over one on EASY only 55%. 480 noisy verdicts cannot pin
               gaps that small, so some mass stays on three searches, and on one.
  1 epoch      DPO has no ratio to theta_old and no clip: its loss does not care who sampled the
               pairs, so reusing them needs no protection. One pass is simply 10x less training.
  beta         sets the target. beta 0.5 is held near pi_ref ({results['beta 0.5'][-1]:.3f}, target {targets[0.5]:.3f}); its
               target is barely above 0.7, so its "stuck" runs are just runs at their target.
               beta 0.02 aims at {targets[0.02]:.3f} but gets no further than beta 0.1 in 60 iterations.
  online       fresher pairs and a slightly higher J, but the implicit reward drifts from r (r err
               {errors['online']:.2f} vs {errors['DPO']:.2f}), and beta stops holding the policy at pi*_beta: at beta 0.5
               it ends above the target ({results['online, b 0.5'][-1]:.3f} vs {targets[0.5]:.3f}).
  by outcome   works from pi_ref's pairs ({results['by outcome'][-1]:.3f}): many of them are two searches against
               none, which the coin always decides the right way. Online it gets stuck
               ({results['outcome, onl.'][-1]:.3f}): once the policy searches once on HARD, every pair left is a coin
               flip. One lucky search (0.9) beats two (0.8) half the time, and loses to none
               (-1.1 vs -1.0) half the time, so nothing pushes it on. DPO keeps only WHICH attempt
               won, never by how much.
  GRPO G 2     the same failure, one module back: divided by the std, a pair's advantages are always
               +0.71 and -0.71, just as DPO's verdict is only a sign ({results['GRPO G 2'][-1]:.3f}). Without the
               divide, Dr.GRPO keeps the size of the gap ({results['GRPO G 2, Dr'][-1]:.3f}).""")

    print("\nThe reward DPO learned, beta * log(pi / pi_ref), relative to 0 searches, averaged over the 20 seeds:")
    print(f"  {'':<6} {'':<13} {'0':>6} {'1':>6} {'2':>6} {'3 searches':>10}")
    for qtype, label in ((env.HARD, "HARD"), (env.EASY, "EASY")):
        lines = (("the rater's r", TRUE[qtype]), ("DPO", rewards["DPO"][qtype]), ("online", rewards["online"][qtype]))
        for i, (name, values) in enumerate(lines):
            print(f"  {label if i == 0 else '':<6} {name:<13} " + " ".join(f"{float(v):>6.2f}" for v in values))
    print("  From pi_ref's pairs, the policy's log-probs ARE the rater's reward, to within the noise:")
    print("  \"your language model is secretly a reward model\". From its own pairs, they are not.")
    print("\nNext: DPO/ -- the same loss at verl's (batch, response_length) shapes, as its online-DPO recipe writes it.")
