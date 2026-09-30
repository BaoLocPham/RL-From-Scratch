"""verl's online DPO one tensor at a time: ``python DPO/steps_dpo.py``."""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.getenv("RL_IMPL") == "scratch":
    sys.path.insert(0, str(HERE / "from_scratch"))
    from dpo import compute_online_dpo_loss, compute_onlinedpo_pref, get_batch_logps  # noqa: E402
else:
    from common import compute_online_dpo_loss, compute_onlinedpo_pref, get_batch_logps  # noqa: E402
from common import dpo_metrics, prepare_dpo_batch  # noqa: E402  (the trainer's and the actor's code, given)

torch.set_printoptions(precision=4, sci_mode=False)

PAD = 0
PROMPT_LEN = 2

print("0. the batch: two prompts, rollout.n = 2, the two responses to a prompt side by side")
prompts = torch.tensor([[1, 2], [PAD, 3]])                  # left-padded, as verl pads prompts
responses = torch.tensor([[4, 1, 2],                        # prompt 0, response a
                          [4, 3, PAD],                      # prompt 0, response b: 2 tokens, then padding
                          [2, 2, 1],                        # prompt 1, response a
                          [1, 2, 2]])                       # prompt 1, response b
input_ids = torch.cat([prompts.repeat_interleave(2, dim=0), responses], dim=1)
response_mask = torch.tensor([[1., 1., 1.], [1., 1., 0.], [1., 1., 1.], [1., 1., 1.]])
token_level_rewards = torch.zeros(4, 3)
token_level_rewards[[0, 1, 2, 3], [2, 1, 2, 2]] = torch.tensor([1.0, 0.0, 0.5, 0.5])   # at the last real token
print("input_ids (prompt | response):\n", input_ids)
print("response_mask:\n", response_mask)
print("token_level_rewards (one score parked on the last real token):\n", token_level_rewards)
print("Rows 0, 1 answer prompt 0; rows 2, 3 answer prompt 1. The pairing is by position,")
print("not by a uid: the trainer repeats each prompt with interleave=True, so pairs are adjacent.")

print("\n1. compute_onlinedpo_pref: which response of each pair is chosen")
preferences = compute_onlinedpo_pref(token_level_rewards, response_mask)
scores = (token_level_rewards * response_mask).sum(-1)
print("scores:", scores.tolist(), "-> as pairs:", scores.view(-1, 2).tolist())
print("argmax per pair:", scores.view(-1, 2).argmax(1).tolist(), "-> preferences:", preferences.tolist())
print("Pair 0: response a (1.0) beats b (0.0). Pair 1 is a TIE (0.5 and 0.5): argmax returns the")
print("first, so response a is 'chosen' by position alone. The recipe keeps tied pairs; step 5")
print("shows what they do to the gradient.")

print("\n2. get_batch_logps: log pi of each whole response, from the logits of prompt + response")
generator = torch.Generator().manual_seed(0)
logits = torch.randn(4, 5, 5, generator=generator)          # (bs, prompt + response, vocab)
labels = input_ids.clone()
labels[:, :PROMPT_LEN] = -100                                # the prompt is not scored
print("labels (the prompt set to -100):\n", labels)
per_position = torch.log_softmax(logits, -1)
row = 0
shifted = [float(per_position[row, t, labels[row, t + 1]]) for t in range(PROMPT_LEN - 1, 4)]
print(f"row {row}: the logits at position t predict the token at t + 1, so logits[:, :-1] meets labels[:, 1:]:")
for t, value in zip(range(PROMPT_LEN - 1, 4), shifted):
    print(f"  logits at position {t} -> log pi(token {int(labels[row, t + 1])} at position {t + 1}) = {value:+.4f}")
logps = get_batch_logps(logits, labels)
print(f"  summed: {sum(shifted):+.4f}   get_batch_logps -> {float(logps[row]):+.4f}")
print("every row:", logps)
print("average_log_prob=True divides by the scored tokens instead:", get_batch_logps(logits, labels, True))
print("DPO sums: it compares whole responses. A mean would score a response by its average token,")
print("so a long answer and a short one with the same per-token odds would look alike.")

print("\n3. prepare_dpo_batch: split into chosen / rejected, and sum the reference over the mask")
ref_logits = torch.randn(4, 5, 5, generator=generator)
ref_log_prob = torch.log_softmax(ref_logits, -1)[:, PROMPT_LEN - 1:-1] \
    .gather(-1, responses.unsqueeze(-1)).squeeze(-1)                     # (bs, 3): what the ref worker returns
batch = prepare_dpo_batch(input_ids, response_mask, ref_log_prob, preferences, PROMPT_LEN)
print("ref_log_prob, per response token:\n", ref_log_prob)
print("reference_chosen_logps  ", batch["reference_chosen_logps"], " rows", preferences.nonzero().flatten().tolist())
print("reference_rejected_logps", batch["reference_rejected_logps"], " rows", (~preferences).nonzero().flatten().tolist())
masked = labels.clone()
masked[1, -1] = -100
recipe, with_pad_masked = float(get_batch_logps(logits, labels)[1]), float(get_batch_logps(logits, masked)[1])
print(f"Row 1 has a padded position. Its labels keep the PAD token, so the policy's sum counts it:")
print(f"  get_batch_logps(row 1) = {recipe:+.4f} with the recipe's labels, {with_pad_masked:+.4f} with the pad also -100.")
print("The reference's sum, over response_mask, never counts it. In the recipe as written the two")
print("sides of the log-ratio can cover different tokens when responses are padded.")

print("\n4. compute_online_dpo_loss: four log-probs per pair -> one number")
policy_chosen = get_batch_logps(logits[preferences], labels[preferences])
policy_rejected = get_batch_logps(logits[~preferences], labels[~preferences])
reference_chosen, reference_rejected = batch["reference_chosen_logps"], batch["reference_rejected_logps"]
pi_logratios = policy_chosen - policy_rejected
ref_logratios = reference_chosen - reference_rejected
dpo_logits = pi_logratios - ref_logratios
print("pi_logratios  = log pi(chosen) - log pi(rejected):        ", pi_logratios)
print("ref_logratios = the same under pi_ref:                    ", ref_logratios)
print("logits        = pi_logratios - ref_logratios:             ", dpo_logits)
beta = 0.1
for name, kwargs in (("sigmoid (DPO)", {}), ("sigmoid, label_smoothing 0.1", {"label_smoothing": 0.1}),
                     ("ipo", {"loss_type": "ipo"}), ("sigmoid, reference_free", {"reference_free": True})):
    loss = compute_online_dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta,
                                   **kwargs)
    print(f"  {name:<30} loss = {float(loss):.4f}")
print(f"sigmoid:         mean(-log sigmoid({beta} * logits))")
print(f"label_smoothing: 0.9 * that + 0.1 * mean(-log sigmoid(-{beta} * logits)): some labels may be flipped")
print(f"ipo:             mean((logits - 1 / (2 * {beta}))^2) = mean((logits - 5)^2): a target margin, not 'more'")
print("reference_free:  ref_logratios replaced by 0: nothing anchors the policy to pi_ref")

print("\n5. what a tie does: the gradient of the sigmoid loss")
print("  d loss / d log pi(chosen) = -beta * sigmoid(-beta * logits) = -(d loss / d log pi(rejected))")
same = torch.tensor([-2.0], requires_grad=True)
loss = compute_online_dpo_loss(same, same, torch.tensor([-2.5]), torch.tensor([-2.5]), beta)
loss.backward()
print(f"  the SAME response twice: logits 0, loss {float(loss.detach()):.4f}, gradient on its log-prob {float(same.grad):+.4f}")
chosen = torch.tensor([-2.0], requires_grad=True)
rejected = torch.tensor([-2.0], requires_grad=True)
loss = compute_online_dpo_loss(chosen, rejected, torch.tensor([-2.5]), torch.tensor([-2.5]), beta)
loss.backward()
print(f"  two DIFFERENT responses that tied: gradient {float(chosen.grad):+.4f} on the first, {float(rejected.grad):+.4f} on the second")
print("  Identical responses cancel exactly, like GRPO's zero-variance group. Different responses")
print("  with the same score still get pushed apart, toward whichever came first: noise.")

print("\n6. the metrics dp_actor logs, from batch means")
metrics = dpo_metrics(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
for key, value in metrics.items():
    print(f"  {key:<20} {value:+.4f}")
per_pair = float(((beta * (policy_chosen - reference_chosen)) > (beta * (policy_rejected - reference_rejected))).float().mean())
print(f"  rewards_accuracies is one bit from the averaged log-probs; per pair, as TRL counts it: {per_pair:.2f}")
print("  rewards_chosen = beta * (log pi - log pi_ref): the implicit reward. The margin is what the")
print("  loss pushes; the two can both rise or both fall.")

print("\n7. ref_update_freq: where pi_ref comes from")
print("  The recipe's config sets ref_update_freq: 1, copying the actor into the reference model")
print("  every step. Then pi_ref = the policy that generated the batch, ref_logratios = pi_logratios")
sync = compute_online_dpo_loss(policy_chosen, policy_rejected, policy_chosen.detach(), policy_rejected.detach(), beta)
print(f"  at the start of the update, logits = 0 and the loss is log 2 = {float(sync):.4f} for every batch.")
print("  DPO against your previous self: SPIN's game. ref_update_freq: -1 keeps pi_ref frozen, as")
print("  the published, offline DPO does.")
