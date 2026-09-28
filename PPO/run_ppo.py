"""Train the token task with PPO: ``python PPO/run_ppo.py`` (add ``verl`` to compare verl's version).

The PPO paper's Algorithm 1 on PPO/task.py: 32 responses per iteration, 4
epochs of minibatches of 8 on eq. 9, a critic and GAE -- core PPO, Part 1 of
the exercise. With ``verl``, the same loop is then run with verl's functions
(Part 2): whitened advantages, clipped critic, dual clip, and a KL to a frozen
reference folded into the reward.

``RL_IMPL=scratch`` runs your PPO/from_scratch/ppo.py (and ppo_verl.py, with ``verl``) instead
of the reference.
"""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))
    import ppo as impl                                 # your Part 1
else:
    import common as impl                              # the reference
import task  # noqa: E402

WITH_VERL = "verl" in sys.argv[1:]

if __name__ == "__main__":
    torch.set_num_threads(1)                           # tiny model: one thread is fastest
    mask, _ = task.response_mask_for(2)
    print(f"""The token task: guess a hidden {task.RESPONSE_LENGTH}-token answer {task.TARGET.tolist()}, vocabulary of {task.VOCAB}.
The reward (fraction correct) lands on the LAST real token only. Two response lengths,
so response_mask has two shapes: {mask[0].tolist()} and {mask[1].tolist()}.

Core PPO. Each iteration: 32 new responses from theta_old -> GAE -> 4 epochs x 4 minibatches
of 8 = 16 optimizer steps on eq. 9 -> theta_old <- theta.
""")
    history, model = task.train(impl, every=5)

    print(f"""
Columns (each is the mean over that iteration's 16 optimizer steps):
  reward    fraction of tokens right, averaged over the 32 responses. The real score.
  pg_loss   -L^CLIP. Its value says little -- at theta_old it is just -mean(A); its gradient
            is what moves the policy.
  vf_loss   the critic's error against the GAE returns. Falls as V learns what to expect.
  entropy   S. Starts at ln 3 = 1.0986 (uniform) and falls as the policy commits.

Learned token distribution per position (target {task.TARGET.tolist()}):""")
    for position, row in enumerate(model.policy_logits.softmax(-1).detach()):
        note = "" if position < int(task.LENGTHS.min()) else "   <- only the 4-token half of the batch reaches it"
        print(f"  position {position}: {[round(v, 3) for v in row.tolist()]}  argmax {int(row.argmax())}{note}")
    print(f"""
Critic V(s_t) per position: {[round(v, 3) for v in model.value_head.detach().tolist()]}
It predicts the final reward from each position: close to 1 once the policy is right.""")

    if WITH_VERL:
        print("\nThe same loop with verl's functions (Part 2 of the exercise):\n")
        vimpl = __import__("ppo_verl") if os.getenv("RL_IMPL") == "scratch" else impl   # your Part 2
        task.train(vimpl, verl=True, every=5)
        print("""
  clipfrac  share of tokens the clip is holding back: largest early, while the policy moves fast.
  ppo_kl    mean(old_log_prob - log_prob): how far each update moved from theta_old, estimated
            from the sampled tokens (TRPO's mean_kl, as a k1 estimate).
  Slightly faster than core PPO here, mostly from whitening the advantages.""")
    else:
        print("\n(./scripts/run_ppo.sh run verl also runs verl's version of the same loop.)")
    print("""
That is the whole algorithm. GRPO/ keeps this loop and the clip but drops the critic:
the advantage comes from comparing several answers to the same question instead.""")
