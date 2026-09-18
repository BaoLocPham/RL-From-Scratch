"""Train a tiny policy with verl's GRPO: ``python GRPO/run_grpo.py``.

The same miniature task as ``PPO/run_ppo.py`` -- guess a hidden four-token
target -- so the two are directly comparable. The difference is the whole
lesson: there is no value head here, no GAE and no value loss. The baseline is
the other samples drawn from the same prompt.

Three prompts, each sampled ``GROUP_SIZE`` times per step. Rows sharing a
prompt share an ``index`` and form one group.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))
    from grpo import (agg_loss, compute_grpo_outcome_advantage,  # noqa: E402
                      compute_policy_loss, kl_penalty)
else:
    sys.path.insert(0, str(HERE))
    from common import (agg_loss, compute_grpo_outcome_advantage,  # noqa: E402
                        compute_policy_loss, kl_penalty)

torch.manual_seed(0)

RESPONSE_LENGTH = 4
VOCAB = 3
PROMPTS = 3
GROUP_SIZE = 8          # verl's rollout.n
PPO_EPOCHS = 2
BETA = 0.02             # GRPO's KL coefficient, applied to the LOSS

# One hidden target per prompt.
TARGETS = torch.tensor([[2, 0, 1, 2], [0, 1, 1, 0], [1, 2, 0, 1]])

policy_logits = torch.zeros(PROMPTS, RESPONSE_LENGTH, VOCAB, requires_grad=True)
reference_logits = torch.zeros(PROMPTS, RESPONSE_LENGTH, VOCAB)
optimizer = torch.optim.Adam([policy_logits], lr=0.1)

BATCH = PROMPTS * GROUP_SIZE
index = np.repeat(np.array([f"p{i}" for i in range(PROMPTS)]), GROUP_SIZE)
rows = torch.arange(PROMPTS).repeat_interleave(GROUP_SIZE)
response_mask = torch.ones(BATCH, RESPONSE_LENGTH)

print(f"{PROMPTS} prompts x rollout.n={GROUP_SIZE} = {BATCH} responses per step.")
print("index:", index.tolist())
print("No value head, no GAE, no value loss -- the group is the baseline.\n")
print(f"{'step':>5}{'reward':>9}{'pg_loss':>10}{'kl':>9}{'clipfrac':>10}"
      f"{'zero-var groups':>17}")

for step in range(60):
    with torch.no_grad():
        old_logits = policy_logits.detach().clone()
        probs = old_logits[rows].softmax(-1)
        tokens = torch.multinomial(probs.reshape(-1, VOCAB), 1).reshape(BATCH, RESPONSE_LENGTH)

        correct = (tokens == TARGETS[rows]).float().mean(-1)
        token_level_rewards = torch.zeros(BATCH, RESPONSE_LENGTH)
        token_level_rewards[:, -1] = correct          # outcome reward, last token

        advantages, _ = compute_grpo_outcome_advantage(
            token_level_rewards.clone(), response_mask, index)

        # A group whose samples all scored the same has zero std, so every
        # advantage in it is zero and it teaches nothing. DAPO's dynamic
        # sampling exists to detect and resample exactly these.
        per_group = correct.reshape(PROMPTS, GROUP_SIZE)
        dead = int((per_group.std(dim=-1) < 1e-6).sum())

        old_log_prob = old_logits[rows].log_softmax(-1) \
            .gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        ref_log_prob = reference_logits[rows].log_softmax(-1) \
            .gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    for epoch in range(PPO_EPOCHS):
        log_prob = policy_logits[rows].log_softmax(-1) \
            .gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

        pg_loss, pg_clipfrac, _, _ = compute_policy_loss(
            old_log_prob, log_prob, advantages, response_mask, cliprange=0.2,
            loss_agg_mode="token-mean")

        # GRPO's KL goes in the LOSS, not in the reward. Contrast PPO/run_ppo.py,
        # which folds it into token_level_rewards before GAE.
        per_token_kl = kl_penalty(log_prob, ref_log_prob, "k3")
        kl = agg_loss(per_token_kl, response_mask, "token-mean")
        loss = pg_loss + BETA * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if step % 10 == 0 or step == 59:
        print(f"{step:>5}{correct.mean():>9.3f}{pg_loss.item():>10.4f}"
              f"{kl.item():>9.4f}{pg_clipfrac.item():>10.2f}{dead:>13} / {PROMPTS}")

print("\nlearned argmax per prompt:")
final = policy_logits.softmax(-1).detach()
for prompt in range(PROMPTS):
    got = final[prompt].argmax(-1)
    want = TARGETS[prompt]
    print(f"  prompt {prompt}: got {got.tolist()}  target {want.tolist()}"
          f"  {'ok' if torch.equal(got, want) else 'MISMATCH'}")

print("\nThe 'zero-var groups' column is worth watching. Once a prompt is solved,")
print("all 8 of its samples score the same, its std is 0, and every advantage in")
print("that group becomes 0 -- it stops contributing gradient entirely. That is")
print("not a failure mode, it is GRPO reaching the end of what a group can say.")
print("DAPO's dynamic sampling detects these groups and resamples instead.")
