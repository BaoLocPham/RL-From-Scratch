"""PPO one stage at a time: ``python PPO/steps_ppo.py`` (or ``... core`` for Part 1 only).

Part 1 is core PPO, the paper's eq. 9 and Algorithm 1 (stages 1-6). Part 2 is
what verl adds on top (stages 7-12). Every equation is printed with the numbers
substituted into it. Set ``RL_IMPL=scratch`` to run the same walkthrough on
your PPO/from_scratch/ppo.py, and ``./scripts/run_ppo.sh diff core`` to compare
Part 1 with the reference line by line (``diff`` alone compares both parts).
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

CORE_ONLY = "core" in sys.argv[1:]


def banner(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def f(x):
    x = x.detach() if isinstance(x, torch.Tensor) else x
    return round(float(x), 4) + 0.0                    # + 0.0: no -0.0


def run(verl):
    history, model = task.train(impl, verl=verl)
    print("  " + task.header(verl))
    for i in (0, 5, 10, 20, 39):
        print("  " + task.row(i, history[i]))
    print(f"  learned tokens {model.policy_logits.argmax(-1).tolist()}, target {task.TARGET.tolist()}")


R = torch.tensor([[0., 0., 1., 0.], [0., -1., 0., 0.]])
V = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.4, 0.3, 0.2]])
M = torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]])
one = torch.ones(1, 1)

print("PART 1 -- core PPO: the paper's eq. 9 and Algorithm 1")

# ---------------------------------------------------------------- stage 1
banner("STAGE 1  from one decision to T tokens: log pi per token, and the masked mean")
logits = torch.tensor([[[0., 0., 0.], [2., 0., -1.]]])
lp = impl.logprobs_from_logits(logits, torch.tensor([[1, 0]]))
print("  The toy track: policy.dist(qtype).log_prob(action) -- one decision, 2 actions.")
print("  An LLM: one row of logits PER POSITION, over the whole vocabulary.")
print("  logits [[0, 0, 0], [2, 0, -1]], sampled tokens [1, 0]")
print(f"  position 0: log(1/3)                             = {f(lp[0, 0]):+.4f}")
print(f"  position 1: 2 - log(e^2 + e^0 + e^-1) = 2 - 2.1698 = {f(lp[0, 1]):+.4f}")
print("  -> shape (batch, response_length): one log-prob per token")
spread, mask4 = torch.tensor([[1., 2., 3., 100.]]), torch.tensor([[1., 1., 1., 0.]])
print("\n  values [1, 2, 3, 100], mask [1, 1, 1, 0]   (the 100 is padding)")
print(f"  plain mean  = {f(spread.mean()):.4f}   <- the padding dominates")
print(f"  masked_mean = (1 + 2 + 3) / 3 = {f(impl.masked_mean(spread, mask4)):.4f}   <- the paper's E_t, over real tokens")

# ---------------------------------------------------------------- stage 2
banner("STAGE 2  GAE: delta_t = r_t + gamma V(s_t+1) - V(s_t);  A_t = delta_t + gamma lam A_t+1")
gamma, lam = 0.9, 0.95
print(f"  gamma {gamma}, lam {lam} (gamma * lam = {gamma * lam:.3f}). Row 0: rewards [0, 0, 1], values [0.1, 0.2, 0.3]")
next_v, carry = 0.0, 0.0
for t in (2, 1, 0):
    delta = float(R[0, t]) + gamma * next_v - float(V[0, t])
    carry = delta + gamma * lam * carry
    print(f"  t={t}: delta = {float(R[0, t]):.0f} + {gamma} * {next_v:.1f} - {float(V[0, t]):.1f} = {delta:+.4f}   "
          f"A_{t} = {carry:.4f}")
    next_v = float(V[0, t])
advantages, returns = impl.compute_gae(R, V, M, gamma, lam)
print(f"  compute_gae: advantages {[f(v) for v in advantages[0, :3]]}, returns = A + V {[f(v) for v in returns[0, :3]]}")
print("  The returns are the critic's target (stage 4): what actually followed each position.")
hole_r, hole_v, hole_m = (torch.tensor([[0., 0., 0., 1.]]), torch.tensor([[0.1, 0.5, 0.2, 0.3]]),
                          torch.tensor([[1., 0., 1., 1.]]))
_, hole_returns = impl.compute_gae(hole_r, hole_v, hole_m, 1.0, 1.0)
print("\n  An agentic response: token 1 is a tool's OUTPUT, mask [1, 0, 1, 1], reward 1 at the end.")
print(f"  returns -> {[f(v) for v in hole_returns[0]]}: token 0 still gets the final reward (1.0), because")
print("  the mask CARRIES the running values across position 1. A `dones`-style reset would cut")
print("  the response in two and give token 0 less.")

# ---------------------------------------------------------------- stage 3
banner("STAGE 3  L^CLIP per token: max(-A r, -A clip(r, 0.8, 1.2)) -- Surrogates' min, negated")
print(f"  {'A':>4} {'r':>5} | {'-A r':>7} {'-A clip(r)':>10} | {'loss':>7}")
for a, r in ((1.0, 1.1), (1.0, 1.35), (1.0, 0.7), (-1.0, 0.7), (-1.0, 1.35)):
    out = impl.ppo_clip_loss(torch.zeros(1, 1), torch.tensor([[r]]).log(), torch.tensor([[a]]), one)
    clipped = min(max(r, 0.8), 1.2)
    print(f"  {a:>+4.0f} {r:>5.2f} | {-a * r:>+7.3f} {-a * clipped:>+10.3f} | {f(out):>+7.3f}")
print("  The larger (worse) of the two is kept: past 1 + eps a good token earns nothing more,")
print("  but a token moved the wrong way pays in full. Over many tokens: the masked mean.")

# ---------------------------------------------------------------- stage 4
banner("STAGE 4  L^VF = 0.5 * (V(s_t) - V_targ)^2, averaged over the mask")
vf = impl.value_loss(torch.tensor([[0.2]]), torch.tensor([[0.75]]), one)
print(f"  one token: prediction 0.2, return 0.75 -> 0.5 * (0.2 - 0.75)^2 = {f(vf):.5f}")
print("  A plain regression: each position's prediction chases the return that followed it.")

# ---------------------------------------------------------------- stage 5
banner("STAGE 5  S: entropy = logsumexp(logits) - sum(softmax(logits) * logits)")
ent = impl.entropy_from_logits(torch.tensor([[[0., 0., 0.], [2., 0., -1.], [9., 0., 0.]]]))
for label, e in zip(("[0, 0, 0]  uniform", "[2, 0, -1]", "[9, 0, 0]  almost certain"), ent[0]):
    print(f"  logits {label:<26} H = {f(e):.4f}")
print("  Eq. 9 ADDS c2 * S to the objective, so the loss SUBTRACTS it: a bonus for staying uncertain,")
print("  so the policy does not collapse onto one token before the reward has spoken.")

# ---------------------------------------------------------------- stage 6
banner("STAGE 6  the loop: Algorithm 1, with eq. 9 in the slot")
print("""  for iteration:
      batch = rollout(model)                                theta_old: 32 responses, 4 tokens each
      batch = compute_advantage(batch)                      stage 2's GAE, once
      for epoch in range(4):                                K = 4
          for minibatch of 8:                               M = 8, so 16 optimizer steps per batch
              loss = pg_loss + 0.5 * vf_loss - 0.01 * S     eq. 9, negated""")
torch.manual_seed(0)
model = task.TokenModel()
batch = impl.compute_advantage(task.rollout(model, 32, impl.logprobs_from_logits))
logits = model.logits(32)
log_prob = impl.logprobs_from_logits(logits, batch["tokens"])
pg = impl.ppo_clip_loss(batch["old_log_prob"], log_prob, batch["advantages"], batch["response_mask"])
vf = impl.value_loss(model.values(32), batch["returns"], batch["response_mask"])
ent = impl.entropy_bonus(logits, batch["response_mask"])
print(f"\n  the first step, whole batch: loss = {f(pg):+.4f} + 0.5 * {f(vf):.4f} - 0.01 * {f(ent):.4f}"
      f" = {f(pg + 0.5 * vf - 0.01 * ent):+.4f}")
print("  (every ratio is 1 at theta_old, so pg_loss is just -mean(A); its gradient is what moves the policy)\n")
run(verl=False)
print("  That is core PPO.")

if CORE_ONLY:
    raise SystemExit(0)

print("\n\nPART 2 -- verl's extras: what production adds on top of eq. 9")

# ---------------------------------------------------------------- stage 7
banner("STAGE 7  whitening: advantages rescaled to mean 0, spread 1, over the mask")
print("  values [1, 2, 3, 100], mask [1, 1, 1, 0]")
print(f"  masked_var  = mean(1, 0, 1) * 3/2 = {f(impl.masked_var(spread, mask4)):.4f}   (n/(n-1), n = real positions)")
print(f"  masked_whiten -> {[f(v) for v in impl.masked_whiten(spread, mask4)[0, :3]]}")
advantages, returns = impl.compute_gae_advantage_return(R, V, M, gamma, lam)
print(f"  verl's GAE, row 0: advantages {[f(v) for v in advantages[0, :3]]} (whitened), returns "
      f"{[f(v) for v in returns[0, :3]]} (untouched)")
print("  Why: without it the policy's step size follows the reward's scale.")

# ---------------------------------------------------------------- stage 8
banner("STAGE 8  E_t over tokens: loss_agg_mode decides what gets equal weight")
loss = torch.tensor([[1., 2., 3., 100.], [4., 5., 100., 100.]])
print("  loss [[1, 2, 3, pad], [4, 5, pad, pad]]: responses of length 3 and 2, width 4")
for mode, how in (("token-mean", "(1+2+3+4+5) / 5"), ("seq-mean-token-sum", "(6 + 9) / 2"),
                  ("seq-mean-token-mean", "(6/3 + 9/2) / 2"), ("seq-mean-token-sum-norm", "(6 + 9) / 4")):
    print(f"  {mode:<24} {how:<18} = {f(impl.agg_loss(loss, M, mode)):.4f}")
print("  Core PPO's token-mean lets the longer response count for more; seq-mean-token-mean makes")
print("  length irrelevant; sum-norm divides by a constant (Dr.GRPO).")

# ---------------------------------------------------------------- stage 9
banner("STAGE 9  verl's policy loss: asymmetric range, dual clip, and metrics")
out = impl.compute_policy_loss(torch.zeros(1, 1), torch.tensor([[3.5]]).log(), -one, one, cliprange=0.2)[0]
print(f"  one token A = -1, r = 3.5: core clip max(3.5, 1.2) = 3.5; dual clip min(3.5, 3.0) = {f(out):.4f}")
old = torch.zeros(2, 4)
logp = torch.tensor([[0.1, -0.1, 0.3, 0.], [0.8, 0.9, 0., 0.]])
adv = torch.tensor([[1., 1., 1., 0.], [-1., -1., 0., 0.]])
for c in (3.0, 1.5):
    pg, clipfrac, ppo_kl, lower = impl.compute_policy_loss(old, logp, adv, M, cliprange=0.2, clip_ratio_c=c)
    print(f"  batch, clip_ratio_c {c}: pg_loss {f(pg):+.4f}  clipfrac {f(clipfrac):.2f}  ppo_kl {f(ppo_kl):+.4f}"
          f"  clipfrac_lower {f(lower):.2f}")
for low, high in ((0.2, 0.2), (0.2, 0.3)):
    pg = impl.compute_policy_loss(old, logp, adv, M, cliprange=0.2, cliprange_low=low, cliprange_high=high)[0]
    print(f"  range [1-{low}, 1+{high}]: pg_loss {f(pg):+.4f}")
print("  clipfrac and ppo_kl are how you read whether the policy moves too fast; ppo_kl is the k1")
print("  estimate of TRPO's exact mean_kl.")

# ---------------------------------------------------------------- stage 10
banner("STAGE 10  verl's value loss: clipped around the critic's OLD prediction")
vf, _ = impl.compute_value_loss(torch.tensor([[0.2]]), torch.tensor([[0.75]]), torch.tensor([[0.1]]), one, 0.05)
print("  one token: old 0.1, target 0.75, new 0.2, cliprange 0.05 -> clipped prediction 0.15")
print(f"  max((0.2 - 0.75)^2, (0.15 - 0.75)^2) = 0.36 -> 0.5 * 0.36 = {f(vf):.4f}   (core: 0.15125)")

# ---------------------------------------------------------------- stage 11
banner("STAGE 11  KL to a frozen REFERENCE model, folded into the reward (RLHF, not the paper)")
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

# ---------------------------------------------------------------- stage 12
banner("STAGE 12  the same loop with verl's functions")
run(verl=True)
print("  Slightly faster than core PPO on this task, mostly from whitening. The algorithm is the same.")
