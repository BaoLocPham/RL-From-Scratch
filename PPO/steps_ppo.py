"""PPO the way verl writes it, one stage at a time: ``python PPO/steps_ppo.py``.

Follows the exercise's eight stages -- eq. 9's three terms, GAE, and the loop --
printing each equation and the numbers substituted into it. Set
``RL_IMPL=scratch`` to run the same walkthrough on your PPO/from_scratch/ppo.py,
and ``./scripts/run_ppo.sh diff`` to compare the two outputs line by line.
"""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                          # task.py, always given
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))
    import ppo as impl                                 # your implementation
else:
    import common as impl                              # the reference
import task  # noqa: E402

torch.set_printoptions(precision=4, sci_mode=False)


def banner(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def f(x):
    x = x.detach() if isinstance(x, torch.Tensor) else x
    return round(float(x), 4) + 0.0                    # + 0.0: no -0.0


R = torch.tensor([[0., 0., 1., 0.], [0., -1., 0., 0.]])
V = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.4, 0.3, 0.2]])
M = torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]])

# ---------------------------------------------------------------- stage 1
banner("STAGE 1  from one decision to T tokens: log pi per token, and the mask")
logits = torch.tensor([[[0., 0., 0.], [2., 0., -1.]]])
labels = torch.tensor([[1, 0]])
lp = impl.logprobs_from_logits(logits, labels)
print("  The toy track: policy.dist(qtype).log_prob(action) -- one decision, 2 actions.")
print("  An LLM: one row of logits PER POSITION, over the whole vocabulary.")
print(f"  logits {logits[0].tolist()}, sampled tokens {labels[0].tolist()}")
print(f"  position 0: log(1/3)                           = {f(lp[0, 0]):+.4f}")
print(f"  position 1: 2 - log(e^2 + e^0 + e^-1) = 2 - 2.1698 = {f(lp[0, 1]):+.4f}")
print(f"  logprobs_from_logits -> {[f(v) for v in lp[0]]}, shape (batch, response_length)")
spread = torch.tensor([[1., 2., 3., 100.]])
mask4 = torch.tensor([[1., 1., 1., 0.]])
print(f"\n  values {spread[0].tolist()}, mask {mask4[0].tolist()}   (the 100 is padding)")
print(f"  plain mean  = {f(spread.mean()):.4f}   <- the padding dominates")
print(f"  masked_mean = (1 + 2 + 3) / 3 = {f(impl.masked_mean(spread, mask4)):.4f}")
print(f"  masked_var  = mean(1, 0, 1) * 3/2 = {f(impl.masked_var(spread, mask4)):.4f}   (n/(n-1), n = real positions)")
print(f"  masked_whiten -> {[f(v) for v in impl.masked_whiten(spread, mask4)[0, :3]]} on the real positions")

# ---------------------------------------------------------------- stage 2
banner("STAGE 2  GAE: delta_t = r_t + gamma V(s_t+1) - V(s_t);  A_t = delta_t + gamma lam A_t+1")
gamma, lam = 0.9, 0.95
print(f"  gamma {gamma}, lam {lam} (gamma * lam = {gamma * lam:.3f}). Row 0: rewards [0, 0, 1], "
      "values [0.1, 0.2, 0.3]")
next_v, carry = 0.0, 0.0
for t in (2, 1, 0):
    delta = float(R[0, t]) + gamma * next_v - float(V[0, t])
    carry = delta + gamma * lam * carry
    print(f"  t={t}: delta = {float(R[0, t]):.0f} + {gamma} * {next_v:.1f} - {float(V[0, t]):.1f} = {delta:+.4f}   "
          f"A_{t} = {carry:.4f}")
    next_v = float(V[0, t])
advantages, returns = impl.compute_gae_advantage_return(R, V, M, gamma, lam)
print(f"  returns = A + V -> {[f(v) for v in returns[0, :3]]}   (the critic's target, stage 5)")
print(f"  advantages, whitened over the mask -> {[f(v) for v in advantages[0, :3]]}")
hole_r, hole_v, hole_m = (torch.tensor([[0., 0., 0., 1.]]), torch.tensor([[0.1, 0.5, 0.2, 0.3]]),
                          torch.tensor([[1., 0., 1., 1.]]))
_, hole_returns = impl.compute_gae_advantage_return(hole_r, hole_v, hole_m, 1.0, 1.0)
print(f"\n  An agentic response: token 1 is a tool's OUTPUT, mask {hole_m[0].tolist()}, reward 1 at the end.")
print(f"  returns -> {[f(v) for v in hole_returns[0]]}: token 0 still gets the final reward (1.0), because")
print("  the mask CARRIES the running values across position 1. A `dones`-style reset would cut")
print("  the response in two and give token 0 nothing.")

# ---------------------------------------------------------------- stage 3
banner("STAGE 3  E_t over tokens: loss_agg_mode decides what gets equal weight")
loss = torch.tensor([[1., 2., 3., 100.], [4., 5., 100., 100.]])
print("  loss [[1, 2, 3, pad], [4, 5, pad, pad]]: responses of length 3 and 2, width 4")
for mode, how in (("token-mean", "(1+2+3+4+5) / 5"), ("seq-mean-token-sum", "(6 + 9) / 2"),
                  ("seq-mean-token-mean", "(6/3 + 9/2) / 2"), ("seq-mean-token-sum-norm", "(6 + 9) / 4")):
    print(f"  {mode:<24} {how:<18} = {f(impl.agg_loss(loss, M, mode)):.4f}")
print("  token-mean lets the longer response count for more; seq-mean-token-mean makes length")
print("  irrelevant; sum-norm divides by a constant (Dr.GRPO).")

# ---------------------------------------------------------------- stage 4
banner("STAGE 4  L^CLIP per token: max(-A r, -A clip(r, 1-eps, 1+eps)) -- Surrogates' min, negated")
print(f"  {'A':>4} {'r':>5} | {'-A r':>7} {'-A clip(r)':>10} | {'loss':>7}")
one = torch.ones(1, 1)
for a, r in ((1.0, 1.1), (1.0, 1.35), (1.0, 0.7), (-1.0, 0.7), (-1.0, 1.35)):
    out = impl.compute_policy_loss(torch.zeros(1, 1), torch.tensor([[r]]).log(), torch.tensor([[a]]), one,
                                   cliprange=0.2)[0]
    clipped = min(max(r, 0.8), 1.2)
    print(f"  {a:>+4.0f} {r:>5.2f} | {-a * r:>+7.3f} {-a * clipped:>+10.3f} | {f(out):>+7.3f}")
print("  The larger (worse) of the two is kept: past 1 + eps the good token earns nothing more,")
print("  but a token moved the wrong way pays in full.")
old = torch.zeros(2, 4)
logp = torch.tensor([[0.1, -0.1, 0.3, 0.], [0.8, 0.9, 0., 0.]])
adv = torch.tensor([[1., 1., 1., 0.], [-1., -1., 0., 0.]])
for c in (3.0, 1.5):
    pg, clipfrac, ppo_kl, lower = impl.compute_policy_loss(old, logp, adv, M, cliprange=0.2, clip_ratio_c=c)
    print(f"  batch, clip_ratio_c {c}: pg_loss {f(pg):+.4f}  clipfrac {f(clipfrac):.2f}  ppo_kl {f(ppo_kl):+.4f}"
          f"  clipfrac_lower {f(lower):.2f}")
print("  Row 1 has A < 0 and a ratio above 2.2. At clip_ratio_c 1.5 the DUAL clip floors its loss")
print("  (clipfrac_lower 0.40); the paper's clip alone would let it grow without bound.")

# ---------------------------------------------------------------- stage 5
banner("STAGE 5  L^VF: 0.5 * max((V - V_targ)^2, (clip(V, V_old +- c) - V_targ)^2)")
vf, _ = impl.compute_value_loss(torch.tensor([[0.2]]), torch.tensor([[0.75]]), torch.tensor([[0.1]]), one, 0.05)
print("  one token: old prediction 0.1, target (return) 0.75, new prediction 0.2, cliprange 0.05")
print("  clipped prediction = clamp(0.2, 0.1 - 0.05, 0.1 + 0.05) = 0.15")
print(f"  unclipped (0.2 - 0.75)^2 = 0.3025, clipped (0.15 - 0.75)^2 = 0.36, keep 0.36 -> 0.5 * 0.36 = {f(vf):.4f}")
print("  The paper's L^VF is just the squared error; verl clips it like the policy, around the")
print("  critic's OLD prediction, so one minibatch cannot yank the critic far.")

# ---------------------------------------------------------------- stage 6
banner("STAGE 6  S: entropy = logsumexp(logits) - sum(softmax(logits) * logits)")
ent = impl.entropy_from_logits(torch.tensor([[[0., 0., 0.], [2., 0., -1.], [9., 0., 0.]]]))
for row, e in zip(("[0, 0, 0]  uniform", "[2, 0, -1]", "[9, 0, 0]  almost certain"), ent[0]):
    print(f"  logits {row:<26} H = {f(e):.4f}")
print("  Eq. 9 ADDS c2 * S to the objective, so the loss SUBTRACTS it: a bonus for staying uncertain,")
print("  so the policy does not collapse onto one token before the reward has spoken.")

# ---------------------------------------------------------------- stage 7
banner("STAGE 7  KL to a frozen REFERENCE model, folded into the reward (not in the paper)")
print("  The paper's eq. 8 KL measures distance from theta_old. RLHF adds a second one: distance from")
print("  the model BEFORE RL, so the policy cannot drift into text that games the reward.")
logprob, ref = torch.tensor([[-1.0, -0.5, -2.0]]), torch.tensor([[-1.2, -0.4, -1.0]])
print("  logprob [-1.0, -0.5, -2.0], ref [-1.2, -0.4, -1.0], r = logprob - ref = [0.2, -0.1, -1.0]")
for kind in ("k1", "abs", "k2", "k3"):
    row = impl.kl_penalty(logprob, ref, kind)
    flag = "" if bool((row >= 0).all()) else "   <- negative on a real token"
    print(f"  {kind:<4} {[f(v) for v in row[0]]}{flag}")
scored = impl.compute_rewards(torch.tensor([[0., 0., 1.]]), torch.zeros(1, 3), torch.full((1, 3), -0.1), 0.2)
print(f"  compute_rewards: score [0, 0, 1] - 0.2 * (0 - (-0.1)) -> {[f(v) for v in scored[0]]}")
print("  The penalty lands on every token, and GAE carries it backwards like any reward.")

# ---------------------------------------------------------------- stage 8
banner("STAGE 8  the loop: Algorithm 1, with eq. 9 in the slot")
print("""  for iteration:
      batch = rollout(model)                                theta_old: 32 responses, 4 tokens each
      batch = compute_advantage(batch)                      stage 7's KL into the reward, stage 2's GAE
      for epoch in range(4):                                K = 4
          for minibatch of 8:                               M = 8, so 16 optimizer steps per batch
              loss = pg_loss + 0.5 * vf_loss - 0.01 * S     eq. 9, negated""")
torch.manual_seed(0)
model = task.TokenModel()
batch = impl.compute_advantage(task.rollout(model, 32, impl.logprobs_from_logits))
logits = model.logits(32)
log_prob = impl.logprobs_from_logits(logits, batch["tokens"])
pg, _, _, _ = impl.compute_policy_loss(batch["old_log_prob"], log_prob, batch["advantages"],
                                       batch["response_mask"], cliprange=0.2)
vf, _ = impl.compute_value_loss(model.values(32), batch["returns"], batch["values"], batch["response_mask"], 0.2)
ent = impl.compute_entropy_loss(logits, batch["response_mask"])
total = pg + 0.5 * vf - 0.01 * ent
print(f"\n  the first iteration's first step, whole batch: pg_loss {f(pg):+.4f}. At theta_old every ratio is 1")
print("  and the whitened advantages average 0, so the VALUE is 0 -- its gradient is not.")
print(f"  loss = {f(pg):+.4f} + 0.5 * {f(vf):.4f} - 0.01 * {f(ent):.4f} = {f(total):+.4f}")
print("\n  " + task.HEADER)
history, model = task.train(impl, iterations=40)
for i in (0, 5, 10, 20, 39):
    m = history[i]
    print(f"  {i:>9} {m['reward']:>7.3f} {m['pg_loss']:>9.4f} {m['vf_loss']:>8.4f} "
          f"{m['entropy']:>8.4f} {m['pg_clipfrac']:>9.3f} {m['ppo_kl']:>8.4f}")
print(f"  learned tokens {model.policy_logits.argmax(-1).tolist()}, target {task.TARGET.tolist()}")
print("  ./scripts/run_ppo.sh run follows the same run in full, with what each column means.")
