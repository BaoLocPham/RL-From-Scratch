"""Train a tiny policy with verl's online DPO: ``python DPO/run_dpo.py``.

The same miniature task as ``PPO/run_ppo.py`` and ``GRPO/run_grpo.py`` -- guess
a hidden four-token target -- so the three are directly comparable. GRPO
turned each group's rewards into advantages. DPO keeps only which of two
responses scored higher, and fits the policy to that verdict:

    rollout.n = 2 responses per prompt -> compute_onlinedpo_pref -> prepare_dpo_batch
    -> get_batch_logps (chosen, rejected) -> compute_online_dpo_loss -> one optimizer step

Three prompts, each repeated 4 times per step (12 pairs, 24 responses: GRPO's
batch). The sequence is [prompt token, 4 response tokens]; the "model" is a
table of logits per prompt and position, as in GRPO/run_grpo.py.
"""

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

RESPONSE_LENGTH = 4
VOCAB = 3
PROMPTS = 3
COPIES = 4              # each prompt 4 times per step: 12 pairs
TARGETS = torch.tensor([[2, 0, 1, 2], [0, 1, 1, 0], [1, 2, 0, 1]])
PROMPT_LEN = 1          # one prompt token; labels[:, :1] = -100


def sequence_logits(table, prompts):
    """(n, 1 + RESPONSE_LENGTH, VOCAB): the logits at position t predict token t + 1.

    Position 0 is the prompt token, whose logits predict response token 0; the
    last position predicts nothing and is dropped by get_batch_logps's shift.
    """
    logits = table[prompts]
    return torch.cat([logits, torch.zeros(len(prompts), 1, VOCAB)], dim=1)


def train(seed=0, steps=60, beta=0.1, lr=0.1, ref_update_freq=0, loss_type="sigmoid", label_smoothing=0.0,
          every=None):
    """The recipe's loop on the token task. ref_update_freq=0 keeps pi_ref frozen; 1 is the recipe's config."""
    torch.manual_seed(seed)
    policy_logits = torch.zeros(PROMPTS, RESPONSE_LENGTH, VOCAB, requires_grad=True)
    reference_logits = torch.zeros(PROMPTS, RESPONSE_LENGTH, VOCAB)
    optimizer = torch.optim.Adam([policy_logits], lr=lr)
    prompts = torch.arange(PROMPTS).repeat_interleave(COPIES).repeat_interleave(2)   # rows 2i, 2i + 1: one pair
    bs = prompts.shape[0]
    response_mask = torch.ones(bs, RESPONSE_LENGTH)
    history = []
    for step in range(steps):
        with torch.no_grad():
            probs = policy_logits[prompts].softmax(-1)
            tokens = torch.multinomial(probs.reshape(-1, VOCAB), 1).reshape(bs, RESPONSE_LENGTH)
            correct = (tokens == TARGETS[prompts]).float().mean(-1)
            token_level_rewards = torch.zeros(bs, RESPONSE_LENGTH)
            token_level_rewards[:, -1] = correct                                        # outcome reward, last token

            preferences = compute_onlinedpo_pref(token_level_rewards, response_mask)    # (bs,) bool
            scores = correct.view(-1, 2)
            ties = int((scores[:, 0] == scores[:, 1]).sum())                            # labelled anyway: first wins

            input_ids = torch.cat([prompts[:, None], tokens], dim=1)                    # (bs, 1 + R)
            ref_log_prob = reference_logits[prompts].log_softmax(-1) \
                .gather(-1, tokens.unsqueeze(-1)).squeeze(-1)                           # (bs, R), per token
            batch = prepare_dpo_batch(input_ids, response_mask, ref_log_prob, preferences, PROMPT_LEN)

        policy_chosen_logps = get_batch_logps(sequence_logits(policy_logits, prompts[preferences]),
                                              batch["chosen_labels"])
        policy_rejected_logps = get_batch_logps(sequence_logits(policy_logits, prompts[~preferences]),
                                                batch["rejected_labels"])
        loss = compute_online_dpo_loss(policy_chosen_logps, policy_rejected_logps,
                                       batch["reference_chosen_logps"], batch["reference_rejected_logps"],
                                       beta, label_smoothing, loss_type)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ref_update_freq and (step + 1) % ref_update_freq == 0:                      # the recipe's reference sync
            reference_logits = policy_logits.detach().clone()

        metrics = dpo_metrics(policy_chosen_logps, policy_rejected_logps, batch["reference_chosen_logps"],
                              batch["reference_rejected_logps"], beta)
        metrics.update(reward=float(correct.mean()), dpo_loss=float(loss.detach()), ties=ties)
        history.append(metrics)
        if every and (step % every == 0 or step == steps - 1):
            print(f"{step:>5}{metrics['reward']:>8.3f}{metrics['dpo_loss']:>10.4f}{metrics['rewards_chosen']:>10.3f}"
                  f"{metrics['rewards_rejected']:>10.3f}{metrics['rewards_margins']:>9.3f}"
                  f"{metrics['rewards_accuracies']:>6.0f}{ties:>8} / {bs // 2}")
    return history, policy_logits.detach()


def kl_to_uniform(policy_logits):
    """KL(pi || pi_ref) per response, summed over its 4 positions, averaged over the prompts; pi_ref is uniform."""
    logp = policy_logits.log_softmax(-1)
    return float((logp.exp() * (logp + torch.log(torch.tensor(float(VOCAB))))).sum(-1).sum(-1).mean())


if __name__ == "__main__":
    print(f"{PROMPTS} prompts x {COPIES} copies x rollout.n=2 = {PROMPTS * COPIES} pairs per step. beta 0.1, "
          "pi_ref frozen, Adam lr 0.1.")
    print("rewards_*: beta * (mean log pi - mean log pi_ref), the implicit reward; acc: 1 if the mean logits > 0.")
    print("ties: pairs whose two responses scored the same, labelled anyway (argmax takes the first).\n")
    print(f"{'step':>5}{'reward':>8}{'dpo_loss':>10}{'r_chosen':>10}{'r_reject':>10}{'margin':>9}{'acc':>6}"
          f"{'ties':>8}")
    history, policy_logits = train(every=10)

    print("\nlearned argmax per prompt:")
    final = policy_logits.softmax(-1)
    for prompt in range(PROMPTS):
        got = final[prompt].argmax(-1)
        want = TARGETS[prompt]
        print(f"  prompt {prompt}: got {got.tolist()}  target {want.tolist()}"
              f"  {'ok' if torch.equal(got, want) else 'MISMATCH'}")

    print("""
Watch rewards_chosen and rewards_rejected: BOTH rise. DPO only asks that chosen
rise MORE than rejected -- the margin -- and in this task the rejected responses
are mostly near-misses that share tokens with the target, so they go up too.
At the end nearly every pair ties: both responses are the target, the pair is
the same response twice, logits = 0 and the loss is log 2 = 0.693 with zero
gradient. That is GRPO's zero-variance group, in pairs.""")

    SEEDS = 10
    print(f"\nThe recipe's knobs, {SEEDS} seeds, 60 steps each:")
    print(f"  {'run':<28}{'reward@20':>10}{'reward@60':>10}{'p(target)':>11}{'KL to pi_ref':>14}")
    rows = {
        "DPO, pi_ref frozen": {},
        "ref_update_freq 1 (config)": {"ref_update_freq": 1},
        "label_smoothing 0.1": {"label_smoothing": 0.1},
        "loss_type ipo": {"loss_type": "ipo"},
        "beta 0.5": {"beta": 0.5},
    }
    for name, kwargs in rows.items():
        runs = [train(seed, **kwargs) for seed in range(SEEDS)]
        at20 = sum(h[19]["reward"] for h, _ in runs) / SEEDS
        at60 = sum(h[-1]["reward"] for h, _ in runs) / SEEDS
        p_target = sum(float(p.softmax(-1).gather(-1, TARGETS.unsqueeze(-1)).mean()) for _, p in runs) / SEEDS
        kl = sum(kl_to_uniform(p) for _, p in runs) / SEEDS
        print(f"  {name:<28}{at20:>10.3f}{at60:>10.3f}{p_target:>11.3f}{kl:>14.2f}")
    print(f"""  (KL to the starting, uniform policy, per 4-token response; a deterministic response is {4 * torch.log(torch.tensor(3.0)):.2f}.)

Every variant learns this task. The checker is exact, so a verdict is never
wrong, only sometimes a tie: there is nothing to be robust to. What differs is
how far each lets the policy go from pi_ref. IPO aims at a fixed margin,
1 / (2 beta), instead of pushing it without end, and beta 0.5 values the anchor
more: both stay softer. Syncing pi_ref to the actor every step (the recipe's
config) means every update is taken at logits 0, where the weight on each pair
is beta * sigmoid(0) = beta / 2 whatever the pair: a policy gradient that knows
only which response won. SimpleDPO/ shows what happens when the verdicts are
noisy: labelled by a coin-flip outcome, online DPO gets stuck.""")
