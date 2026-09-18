"""verl's PPO one tensor at a time: ``python PPO/steps_ppo.py``."""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))
    from ppo import *  # noqa: F403
else:
    sys.path.insert(0, str(HERE))
    from common import *  # noqa: F403

torch.set_printoptions(precision=4, sci_mode=False)

print("0. the shapes everything uses")
token_level_rewards = torch.tensor([[0., 0., 1., 0.], [0., -1., 0., 0.]])
values = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.4, 0.3, 0.2]])
response_mask = torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]])
print("token_level_rewards:\n", token_level_rewards)
print("response_mask:\n", response_mask)
print("Everything is (batch, response_length). There is no per-sequence scalar:")
print("one outcome reward for a whole response is a row that is zero except at")
print("its last valid position. response_mask is the only thing making a")
print("position real -- row 0 has 3 real tokens, row 1 has 2.")

print("\n1. masked statistics ignore padding entirely")
spread = torch.tensor([[1., 2., 3., 100.]])
mask4 = torch.tensor([[1., 1., 1., 0.]])
print("values:", spread, " mask:", mask4)
print(f"plain mean : {spread.mean():.4f}   <- the padded 100. dominates")
print(f"masked_mean: {masked_mean(spread, mask4):.4f}")  # noqa: F405
print(f"masked_var : {masked_var(spread, mask4):.4f}  (Bessel-corrected)")  # noqa: F405
print("masked_whiten:", masked_whiten(spread, mask4))  # noqa: F405
print("Padding must never reach a statistic. Every mean below is this one.")

print("\n2. GAE, and what response_mask does inside it")
advantages, returns = compute_gae_advantage_return(  # noqa: F405
    token_level_rewards, values, response_mask, gamma=0.9, lam=0.95)
print("advantages (whitened):\n", advantages)
print("returns (not whitened):\n", returns)
print("The mask is NOT a `dones` flag. On a masked position the running")
print("nextvalues/lastgaelam are carried through unchanged, so credit flows")
print("across padding and observation tokens to the next real token. Reset")
print("them to zero instead and every response silently gets cut into pieces.")

print("\n3. loss_agg_mode decides what gets equal weight")
loss = torch.tensor([[1., 2., 3., 100.], [4., 5., 100., 100.]])
print("loss matrix (padding is the 100s):\n", loss)
for mode in ("token-mean", "seq-mean-token-sum", "seq-mean-token-mean",
             "seq-mean-token-sum-norm"):
    print(f"  {mode:<26}{float(agg_loss(loss, response_mask, mode)):>9.4f}")  # noqa: F405
print("Row 0 has 3 tokens, row 1 has 2. token-mean lets the longer row count")
print("for more; seq-mean-token-mean makes length irrelevant. That length")
print("coupling is exactly what the Dr.GRPO paper is about, and")
print("seq-mean-token-sum-norm is its fix: a constant divisor, the padded width.")

print("\n4. the policy loss has two clips, not one")
old_log_prob = torch.zeros(2, 4)
# Row 1 pairs a negative advantage with a ratio above 2.2 -- a response the
# policy is now much more likely to emit, which the reward said was bad.
log_prob = torch.tensor([[0.1, -0.1, 0.3, 0.], [0.8, 0.9, 0., 0.]])
adv = torch.tensor([[1., 1., 1., 0.], [-1., -1., 0., 0.]])
print("ratio:\n", torch.exp(log_prob - old_log_prob))
print("advantages (row 0 positive, row 1 negative):\n", adv)
for c in (3.0, 1.5):
    loss, clipfrac, ppo_kl, lower = compute_policy_loss(  # noqa: F405
        old_log_prob, log_prob, adv, response_mask, cliprange=0.2, clip_ratio_c=c)
    print(f"clip_ratio_c={c:<5} loss={float(loss):+.6f}  clipfrac={float(clipfrac):.2f}"
          f"  ppo_kl={float(ppo_kl):+.4f}  clipfrac_lower={float(lower):.2f}")
print("The dual clip only touches positions where the advantage is NEGATIVE and")
print("the ratio has run past clip_ratio_c. There the ordinary clip leaves the")
print("objective unbounded below, so one badly-rated sample can dominate an")
print("update. clipfrac_lower is how you see it engage: 0.00 then 0.40.")
positive = torch.ones(2, 4)
same = [float(compute_policy_loss(old_log_prob, log_prob, positive,  # noqa: F405
                                  response_mask, cliprange=0.2, clip_ratio_c=c)[0])
        for c in (3.0, 1.5)]
print(f"with every advantage positive: {same[0]:+.6f} and {same[1]:+.6f} -- identical,")
print("because the dual clip has nothing to act on.")

print("\n5. asymmetric clipping is one argument, and it is what DAPO changes")
for low, high in ((0.2, 0.2), (0.2, 0.3)):
    out = compute_policy_loss(old_log_prob, log_prob, adv, response_mask,  # noqa: F405
                              cliprange=0.2, cliprange_low=low, cliprange_high=high)
    print(f"  [1-{low}, 1+{high}]  loss={float(out[0]):+.6f}  clipfrac={float(out[1]):.2f}")
print("More room to raise a good-but-unlikely response than to suppress a bad one.")

print("\n6. the value loss clips around the critic's OWN previous prediction")
vpreds = torch.tensor([[0.2, 0.3, 0.4, 0.], [0.6, 0.5, 0., 0.]])
vf_loss, vf_clipfrac = compute_value_loss(vpreds, returns, values,  # noqa: F405
                                          response_mask, cliprange_value=0.05)
print(f"vf_loss={float(vf_loss):.6f}  vf_clipfrac={float(vf_clipfrac):.2f}")
print("`values` is the old baseline, `returns` the target. Keeping the LARGER")
print("squared error is the same pessimism as the policy clip.")

print("\n7. four KL estimators against a frozen reference")
logprob = torch.tensor([[-1.0, -0.5, -2.0, 0.]])
ref_logprob = torch.tensor([[-1.2, -0.4, -1.0, 0.]])
for kind in ("k1", "abs", "k2", "k3"):
    row = kl_penalty(logprob, ref_logprob, kind)  # noqa: F405
    flag = "" if bool((row >= 0).all()) else "   <- negative on a real token"
    print(f"  {kind:<4}{str([round(v, 5) for v in row[0].tolist()]):<40}{flag}")
print("k1 is unbiased but signed: one sample of it is negative about half the")
print("time, while the quantity it estimates never is. k3 is non-negative AND")
print("unbiased, which is why GRPO uses it.")

print("\n8. where the KL goes -- and this differs between PPO and GRPO")
penalised = compute_rewards(token_level_rewards, old_log_prob,  # noqa: F405
                            torch.full((2, 4), -0.1), kl_ratio=0.2)
print("token_level_scores:\n", token_level_rewards)
print("after compute_rewards:\n", penalised)
print("PPO folds the KL into the REWARD, before GAE, so it is bootstrapped and")
print("discounted along the trajectory. GRPO adds its KL to the final loss")
print("instead. Same intent, different placement, not interchangeable.")
