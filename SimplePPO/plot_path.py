"""The whole path on one toy, VPG to PPO: ``python SimplePPO/plot_path.py``.

Every method of the learning path, run on SimplePPO's three-turn toy with the
same budget -- 60 iterations of 16 episodes, SGD lr 0.3, 20 seeds -- so their
curves share one plot. The same style as Surrogates/surrogates.png, with the
full PPO added as the last line:

    VPG, 1 epoch        L^PG, one step per batch                 advantage: episode reward - batch mean
    VPG, 50 epochs      L^PG, the batch reused 50 epochs          "
    TRPO                L^CPI, rolled back past KL 0.01           "
    L^CPI, no clip      the ratio, 50 epochs, no limit             "
    L^CLIP              Surrogates' winner: the clip, 50 epochs   "
    PPO                 L^CLIP + value loss - entropy, 50 epochs  advantage: critic + GAE

Why 50 epochs: at 10, nothing on this small toy overshoots, and reusing the
batch with plain L^PG does as well as PPO (0.860 vs 0.854 at 60 iterations).
The methods only separate once the batch is pushed hard -- the regime
Surrogates/ uses too.

Saves the figure to PPO/ppo_path.png, for the root README. Takes about five minutes.
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "TRPO"))
sys.path.insert(0, str(HERE))
import env  # noqa: E402
import simple_ppo as ppo  # noqa: E402
from trpo import mean_kl  # noqa: E402  (policy.logits[states] works as policy.logits[qtype] did)

SEEDS = 20
ITERATIONS = 60
EPISODES = 16
LR = 0.3
EPOCHS = 50             # epochs per batch for every method that reuses it
OUT = ROOT / "PPO" / "ppo_path.png"


def outcome_advantage(batch):
    """No critic, as on the toy track: every step gets its episode's total reward, minus the batch mean."""
    total = batch["rewards"].sum(1, keepdim=True).expand(-1, env.TURNS)
    return total - total.mean(), total


def ratio_of(policy, steps):
    return torch.exp(policy.dist(steps["states"]).log_prob(steps["actions"]) - steps["old_logp"])


def flat(batch):
    return {key: batch[key].reshape(-1) for key in ("states", "actions", "old_logp", "advantages", "returns")}


def pg_loss(policy, steps):
    return -(policy.dist(steps["states"]).log_prob(steps["actions"]) * steps["advantages"]).mean()


def cpi_loss(policy, steps):
    return -(ratio_of(policy, steps) * steps["advantages"]).mean()


def clip_loss(policy, steps, eps=0.2):
    ratio = ratio_of(policy, steps)
    return -torch.min(ratio * steps["advantages"], ratio.clamp(1 - eps, 1 + eps) * steps["advantages"]).mean()


def minibatch_update(loss_fn, epochs, minibatch_size=16):
    """Epochs of minibatch SGD on a policy-only loss (the critic is not used)."""
    def update(policy, critic, optimizer, batch):
        steps = flat(batch)
        n = steps["states"].shape[0]
        for _ in range(epochs):
            order = torch.randperm(n)
            for start in range(0, n, minibatch_size):
                idx = order[start:start + minibatch_size]
                loss = loss_fn(policy, {key: value[idx] for key, value in steps.items()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return update


def trpo_update(policy, critic, optimizer, batch, max_kl=0.01, epochs=EPOCHS):
    """TRPO as in TRPO/trpo.py: full-batch steps on L^CPI, rolled back once the KL from theta_old passes max_kl."""
    steps = flat(batch)
    old_probs = policy.logits.softmax(-1)[steps["states"]].detach()
    for _ in range(epochs):
        before = policy.logits.detach().clone()
        loss = cpi_loss(policy, steps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if mean_kl(policy, steps["states"], old_probs) > max_kl:
            with torch.no_grad():
                policy.logits.copy_(before)
            break


def method(update, advantage=None):
    """A stand-in for a module with compute_gae and ppo_update, as env.train expects."""
    class Impl:
        compute_gae = staticmethod(ppo.compute_gae)
        ppo_update = staticmethod(update)
    return Impl, advantage


METHODS = {
    "VPG, 1 epoch": method(minibatch_update(pg_loss, epochs=1, minibatch_size=10_000), outcome_advantage),
    "VPG, 50 epochs": method(minibatch_update(pg_loss, EPOCHS), outcome_advantage),
    "TRPO, delta 0.01": method(trpo_update, outcome_advantage),
    "L^CPI, no clip": method(minibatch_update(cpi_loss, EPOCHS), outcome_advantage),
    "L^CLIP (Surrogates)": method(minibatch_update(clip_loss, EPOCHS), outcome_advantage),
    "PPO (clip + critic + GAE)": method(lambda *batch: ppo.ppo_update(*batch, epochs=EPOCHS)),
}


def curves():
    out = {}
    for name, (impl, advantage) in METHODS.items():
        runs = [env.train(impl, seed, iterations=ITERATIONS, episodes=EPISODES, lr=LR, advantage=advantage)[0]
                for seed in range(SEEDS)]
        out[name] = [sum(run[i] for run in runs) / SEEDS for i in range(ITERATIONS)]
        print(f"  {name:<26} J@10 {out[name][9]:.3f}   J@30 {out[name][29]:.3f}   J@60 {out[name][-1]:.3f}",
              flush=True)
    return out


def save_figure(results):
    import matplotlib
    matplotlib.use("Agg")                                 # write a file, no window
    import matplotlib.pyplot as plt

    episodes = [EPISODES * (i + 1) for i in range(ITERATIONS)]
    styles = {"VPG, 1 epoch": ("-", 1.3), "VPG, 50 epochs": ("--", 1.3), "TRPO, delta 0.01": ("-", 1.3),
              "L^CPI, no clip": ("--", 1.3), "L^CLIP (Surrogates)": ("-", 1.6), "PPO (clip + critic + GAE)": ("-", 2.6)}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, curve in results.items():
        style, width = styles[name]
        ax.plot(episodes, curve, style, label=name, linewidth=width)
    ax.axhline(env.BEST_J, color="gray", linestyle=":", linewidth=1, label="best possible")
    ax.set_xlabel("episodes used (3 decisions each)")
    ax.set_ylabel(f"true reward J (mean of {SEEDS} seeds)")
    ax.set_title(f"The whole path on SimplePPO's three-turn toy: same rollouts, {EPOCHS} epochs per batch",
                 fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"\nsaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    torch.set_num_threads(1)                              # tiny model: one thread is fastest
    print(f"Start J {env.true_reward(env.Policy()):.3f}, best {env.BEST_J}. {SEEDS} seeds, "
          f"{ITERATIONS} iterations of {EPISODES} episodes, SGD lr {LR}.\n")
    save_figure(curves())
