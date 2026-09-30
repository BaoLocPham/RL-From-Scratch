"""DPO as verl runs it: the online-DPO loss of its SPIN recipe.

verl's own trainer (`verl/trainer/ppo/core_algos.py`) has no DPO. Its DPO lives
in the SPIN recipe (Self-Play Fine-Tuning, https://arxiv.org/abs/2401.01335),
now in the separate `verl-project/verl-recipe` repository:

    spin/core_algos.py     compute_onlinedpo_pref, compute_online_dpo_loss, get_batch_logps
    spin/spin_trainer.py   fit_dpo: rollout.n = 2 responses per prompt -> pairs -> the DPO batch
    spin/dp_actor.py       update_policy_dpo_with_ref: the loss, the optimizer step, the metrics

Names, argument orders and returns below follow that code. The loop it runs:

    for step:
        generate 2 responses per prompt (rollout.n = 2), score them (token_level_rewards)
        preferences = compute_onlinedpo_pref(...)          the higher-scoring response of each pair
        ref_log_prob from the reference model; summed over response_mask, per response
        loss = compute_online_dpo_loss(get_batch_logps(chosen), get_batch_logps(rejected), ref..., beta)
        one optimizer step
        every ref_update_freq steps: copy the actor's weights into the reference

The recipe's config sets dpo_beta 0.1, rollout.n 2 and ref_update_freq 1: the
reference is the actor as it was one step ago, which is SPIN's "play against
your previous self".

Everything is ``(batch, ...)`` with pairs interleaved: rows 2i and 2i + 1 are
the two responses to prompt i.
"""

import torch
import torch.nn.functional as F


def compute_onlinedpo_pref(token_level_rewards, response_mask):
    """Which response of each pair is chosen: a ``(bs,)`` bool mask, True on the winner.

    Args:
        token_level_rewards: ``(bs, response_length)``, bs = 2 * prompts, the two
            responses to each prompt next to each other.
        response_mask: ``(bs, response_length)``.

    A response's score is its rewards summed over the mask; the pair's winner
    is the ``argmax`` of the two. On a tie, ``argmax`` returns the FIRST: the
    recipe does not drop tied pairs, it labels the first response chosen.
    """
    if token_level_rewards.shape[0] % 2 != 0 or response_mask.shape[0] % 2 != 0:
        raise ValueError(
            f"Input tensor batch dimension must be even for pair comparison, got shapes: "
            f"{token_level_rewards.shape}, {response_mask.shape}"
        )
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError(f"Shape mismatch between rewards {token_level_rewards.shape} and mask {response_mask.shape}")

    scores = (token_level_rewards * response_mask).sum(dim=-1)           # (bs,): one score per response
    score_pairs = scores.view(-1, 2)                                     # (prompts, 2)
    winner_indices = torch.argmax(score_pairs, dim=1)                    # 0 or 1 within each pair; ties -> 0

    num_pairs = score_pairs.shape[0]
    pair_indices = torch.arange(num_pairs, device=scores.device)
    winner_global_indices = (pair_indices * 2) + winner_indices          # the winner's row in the batch

    output_preference_mask = torch.zeros(num_pairs * 2, dtype=torch.bool, device=scores.device)
    output_preference_mask[winner_global_indices] = True
    return output_preference_mask


def get_batch_logps(logits, labels, average_log_prob=False):
    """log pi(labels) per sequence: ``(bs, seq_len, vocab)`` logits, ``(bs, seq_len)`` labels -> ``(bs,)``.

    ``labels`` is the whole sequence, prompt and response, with every position
    that is not scored set to -100 (the recipe sets the prompt's). Two details:

    * **The shift.** The logits at position t predict the token at t + 1, so
      ``logits[:, :-1]`` is paired with ``labels[:, 1:]``.
    * **Sum, not mean.** DPO compares whole responses, so their token log-probs
      are summed. ``average_log_prob=True`` divides by the number of scored
      tokens instead (the recipe passes False).

    The recipe computes the per-token log-prob as a negated cross-entropy with
    ``ignore_index=-100``, which is ``log_softmax`` then ``gather``.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits and labels must have the same shape[:-1]")

    labels = labels.contiguous().to(logits.device)
    shift_logits = logits[..., :-1, :].contiguous()                      # position t predicts token t + 1
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    per_token_logps = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    per_token_logps = per_token_logps.view(shift_logits.size(0), shift_logits.size(1))

    loss_mask = shift_labels != -100
    masked_logps = per_token_logps * loss_mask
    sequence_logps = masked_logps.sum(dim=-1)

    if average_log_prob:
        num_valid_tokens = loss_mask.sum(dim=-1)
        return sequence_logps / torch.clamp(num_valid_tokens, min=1)
    return sequence_logps


def compute_online_dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps,
                            reference_rejected_logps, beta, label_smoothing=0.0, loss_type="sigmoid",
                            reference_free=False):
    """The DPO loss, averaged over the pairs. Every input is ``(pairs,)``.

        logits = (log pi(chosen) - log pi(rejected)) - (log pi_ref(chosen) - log pi_ref(rejected))

    ``loss_type="sigmoid"`` is DPO (Rafailov et al. eq. 7): -log sigmoid(beta * logits).
    ``label_smoothing`` eps > 0 assumes each label is flipped with probability
    eps (conservative DPO): the loss becomes
    -(1 - eps) log sigmoid(beta * logits) - eps log sigmoid(-beta * logits), whose
    minimum is at a finite margin instead of at infinity.

    ``loss_type="ipo"`` is IPO (Azar et al. 2023, https://arxiv.org/abs/2310.12036):
    (logits - 1 / (2 * beta))^2, a squared error to a fixed target margin. DPO
    keeps pushing a pair that is always decided the same way; IPO stops at the
    target.

    ``reference_free=True`` drops pi_ref: ``ref_logratios`` becomes 0.

    The recipe's error message offers "hinge" as well, but the function
    implements only these two.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = torch.zeros_like(pi_logratios)

    logits = pi_logratios - ref_logratios

    if loss_type == "sigmoid":
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    elif loss_type == "ipo":
        losses = (logits - 1 / (2 * beta)) ** 2
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'sigmoid', 'ipo', or 'hinge'.")

    return losses.mean()


# ------------------------------------------------------------ the trainer and actor around them


def prepare_dpo_batch(input_ids, response_mask, ref_log_prob, preferences, prompt_len):
    """The trainer's "Prepare DPO Batch" block (spin_trainer.fit_dpo), as a function.

    Splits the rollout into chosen and rejected halves by the ``(bs,)``
    preference mask, builds the labels (the prompt's positions set to -100),
    and sums the reference model's per-token log-probs over the response:

        reference_*_logps = (ref_log_prob * response_mask).sum(-1)

    Note what the labels do NOT mask: padding after the end of a response.
    The recipe sets only ``labels[:, :prompt_len]`` to -100, so the policy's
    sums from get_batch_logps include the padded positions, while the
    reference's, taken over ``response_mask``, do not. With equal-length
    responses, as in this repo's demo, the two agree.
    """
    not_preferences = ~preferences
    chosen_input_ids, rejected_input_ids = input_ids[preferences], input_ids[not_preferences]
    chosen_labels = chosen_input_ids.clone()
    chosen_labels[:, :prompt_len] = -100
    rejected_labels = rejected_input_ids.clone()
    rejected_labels[:, :prompt_len] = -100
    ref_sequence_logps = (ref_log_prob * response_mask).sum(dim=-1)
    return {"chosen_input_ids": chosen_input_ids, "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids, "rejected_labels": rejected_labels,
            "reference_chosen_logps": ref_sequence_logps[preferences],
            "reference_rejected_logps": ref_sequence_logps[not_preferences]}


def dpo_metrics(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps,
                beta):
    """What dp_actor logs after an update, computed the way it computes them: from batch MEANS.

        rewards_chosen    beta * (mean log pi(chosen) - mean log pi_ref(chosen))      the implicit reward
        rewards_rejected  the same for the rejected responses
        rewards_margins   rewards_chosen - rewards_rejected
        rewards_accuracies  1.0 if the mean logits are > 0, else 0.0

    ``rewards_accuracies`` is a proxy: one bit per batch, from the averaged
    log-probs. TRL's metric of the same name is the fraction of PAIRS whose
    implicit reward ranks chosen above rejected.
    """
    policy_chosen, policy_rejected = float(policy_chosen_logps.detach().mean()), float(policy_rejected_logps.detach().mean())
    reference_chosen, reference_rejected = float(reference_chosen_logps.mean()), float(reference_rejected_logps.mean())
    logits_mean = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    rewards_chosen = beta * (policy_chosen - reference_chosen)
    rewards_rejected = beta * (policy_rejected - reference_rejected)
    return {"dpo_logits": logits_mean, "rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected,
            "rewards_accuracies": float(logits_mean > 0), "rewards_margins": rewards_chosen - rewards_rejected}
