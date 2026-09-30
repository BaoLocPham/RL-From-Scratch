"""verl's online DPO from scratch. Fill the TODOs, then run ``check.py``.

verl's trainer has no DPO. Its DPO lives in the SPIN recipe
(`verl-project/verl-recipe`, `spin/core_algos.py`), as three functions, and
they are this exercise:

    stage 1  compute_onlinedpo_pref    two responses per prompt -> which one is chosen
    stage 2  get_batch_logps           logits of prompt + response -> log pi(response)
    stage 3  compute_online_dpo_loss   four log-probs per pair -> DPO's loss, or IPO's

**Do `SimpleDPO/` first** if you have not: it builds the same loss on the toy,
with the derivation, and shows what beta, online pairs and outcome-ranked pairs
do. This exercise is that one at verl's shapes. Nothing here needs your PPO or
GRPO exercise: DPO keeps none of PPO's loss.

The trainer's code around these (splitting the batch, the reference log-probs,
the metrics) is given, in ``../common.py``: ``prepare_dpo_batch`` and
``dpo_metrics``. Try not to open the rest of that file.
"""

import torch
import torch.nn.functional as F  # noqa: F401  (F.logsigmoid, F.log_softmax)


# ============================================================ STAGE 1
def compute_onlinedpo_pref(token_level_rewards, response_mask):
    """Which response of each pair is chosen: a ``(bs,)`` bool mask, True on the winner.

    Args:
        token_level_rewards: ``(bs, response_length)``, with bs = 2 * prompts and
            the batch INTERLEAVED: rows 2i and 2i + 1 are the two responses to
            prompt i (the trainer repeats prompts with ``interleave=True``).
        response_mask: ``(bs, response_length)``.

    A response's score is its rewards summed over the mask. The winner of a
    pair is the ``argmax`` of its two scores. On a tie ``torch.argmax`` returns
    the FIRST index, so the recipe labels the first response chosen: it does not
    drop ties. Match that.

    Shapes, for 2 prompts (bs = 4, response_length = 3):

        scores                 (4,)     one per response
        score_pairs            (2, 2)   one row per prompt, its two responses
        winner_indices         (2,)     0 or 1: which of the two won
        winner_global_indices  (2,)     the winner's row in the batch: 2 * pair + winner
        returns                (4,)     bool, True at those rows

    Worked example: scores [1.0, 0.0, 0.5, 0.5] -> [True, False, True, False]
    (the second pair is a tie: the first response is chosen).
    """
    if token_level_rewards.shape[0] % 2 != 0 or response_mask.shape[0] % 2 != 0:
        raise ValueError(
            f"Input tensor batch dimension must be even for pair comparison, got shapes: "
            f"{token_level_rewards.shape}, {response_mask.shape}"
        )
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError(f"Shape mismatch between rewards {token_level_rewards.shape} and mask {response_mask.shape}")

    scores = ...                     # TODO stage 1: each response's rewards summed over the mask, (bs,)
    score_pairs = ...                # TODO stage 1: (prompts, 2), one row per pair
    winner_indices = ...             # TODO stage 1: 0 or 1 per pair; ties -> 0

    num_pairs = score_pairs.shape[0]
    pair_indices = torch.arange(num_pairs, device=scores.device)
    winner_global_indices = ...      # TODO stage 1: each winner's row in the whole batch

    output_preference_mask = torch.zeros(num_pairs * 2, dtype=torch.bool, device=scores.device)
    ...                              # TODO stage 1: set the winners' rows to True
    return output_preference_mask


# ============================================================ STAGE 2
def get_batch_logps(logits, labels, average_log_prob=False):
    """log pi(labels) per sequence: ``(bs, seq_len, vocab)`` logits, ``(bs, seq_len)`` labels -> ``(bs,)``.

    ``labels`` is the whole sequence, prompt and response. Positions that are
    not scored hold -100 (the trainer sets the prompt's). Three details:

    * **The shift.** A causal LM's logits at position t predict the token at
      t + 1. So pair ``logits[:, :-1]`` with ``labels[:, 1:]``: both have
      seq_len - 1 positions.
    * **-100 is not a token.** Mask those positions out. If you use
      ``log_softmax`` then ``gather``, replace the -100s by any valid id (0)
      before gathering, or gather will fail; the mask zeroes them afterwards.
      (The recipe uses ``torch.nn.CrossEntropyLoss(ignore_index=-100,
      reduction="none")``, negated: the same numbers.)
    * **Sum, not mean.** DPO compares whole responses. ``average_log_prob=True``
      divides each sum by its number of scored tokens (at least 1) instead.

    Shapes, for bs 4, a 2-token prompt, a 3-token response, vocabulary 5:

        logits            (4, 5, 5)
        shift_logits      (4, 4, 5)   positions 0..3, each predicting the next token
        shift_labels      (4, 4)      positions 1..4
        per_token_logps   (4, 4)      log pi(shift_labels) under shift_logits
        loss_mask         (4, 4)      bool: shift_labels != -100
        returns           (4,)
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits and labels must have the same shape[:-1]")

    shift_logits = ...               # TODO stage 2: every position but the last
    shift_labels = ...               # TODO stage 2: every position but the first
    loss_mask = ...                  # TODO stage 2: True where a label is scored
    per_token_logps = ...            # TODO stage 2: log pi(label) at each shifted position, (bs, seq_len - 1)
    sequence_logps = ...             # TODO stage 2: summed over the scored positions only

    if average_log_prob:
        num_valid_tokens = loss_mask.sum(dim=-1)
        return ...                   # TODO stage 2: the sum over the number of scored tokens, clamped to >= 1
    return sequence_logps


# ============================================================ STAGE 3
def compute_online_dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps,
                            reference_rejected_logps, beta, label_smoothing=0.0, loss_type="sigmoid",
                            reference_free=False):
    """The loss, averaged over the pairs. Every input is ``(pairs,)``; returns ``()``.

        pi_logratios  = log pi(chosen)     - log pi(rejected)
        ref_logratios = log pi_ref(chosen) - log pi_ref(rejected)      0 if reference_free
        logits        = pi_logratios - ref_logratios

    ``loss_type="sigmoid"``: DPO (Rafailov et al. eq. 7), with label smoothing eps
    (the labels are assumed flipped with probability eps):

        losses = -(1 - eps) * logsigmoid(beta * logits) - eps * logsigmoid(-beta * logits)

    ``loss_type="ipo"``: IPO (Azar et al. 2023), a squared error to a fixed
    target margin instead of an ever-larger one:

        losses = (logits - 1 / (2 * beta)) ** 2

    Any other loss_type: raise ValueError. Use F.logsigmoid, not log(sigmoid):
    it stays finite when a pair is badly wrong.

    Worked example, beta 0.1, one pair with logits = 0: sigmoid 0.6931 (log 2),
    ipo (0 - 5)^2 = 25.0.
    """
    pi_logratios = ...               # TODO stage 3
    ref_logratios = ...              # TODO stage 3

    if reference_free:
        ref_logratios = ...          # TODO stage 3: no reference: zeros shaped like pi_logratios

    logits = ...                     # TODO stage 3

    if loss_type == "sigmoid":
        losses = ...                 # TODO stage 3: DPO, with label smoothing
    elif loss_type == "ipo":
        losses = ...                 # TODO stage 3: IPO
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'sigmoid', 'ipo', or 'hinge'.")

    return ...                       # TODO stage 3: the mean over the pairs
