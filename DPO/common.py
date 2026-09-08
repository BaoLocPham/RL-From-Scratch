"""Direct Preference Optimization (DPO) in four pure functions."""

import torch
import torch.nn.functional as F


def sequence_log_probs(logits, token_ids, mask):
    """Sum selected token log-probabilities for each completion.

    ``logits`` is ``(B,T,V)``; ``token_ids`` and ``mask`` are ``(B,T)``.
    Padding contributes nothing. DPO compares whole-completion likelihoods, so
    valid token log-probabilities are summed rather than averaged.
    """
    token_logp = F.log_softmax(logits, dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask.to(token_logp.dtype)).sum(dim=-1)


def preference_logit(policy_chosen_logp, policy_rejected_logp,
                     reference_chosen_logp, reference_rejected_logp, beta=0.1):
    """Return the reference-corrected chosen-vs-rejected margin."""
    chosen_logratios = policy_chosen_logp - reference_chosen_logp
    rejected_logratios = policy_rejected_logp - reference_rejected_logp
    delta_score = chosen_logratios - rejected_logratios
    return beta * delta_score


def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             reference_chosen_logp, reference_rejected_logp, beta=0.1):
    """Mean negative log-sigmoid loss for preference pairs."""
    beta_delta_score = preference_logit(
        policy_chosen_logp, policy_rejected_logp,
        reference_chosen_logp, reference_rejected_logp, beta,
    )
    # ``beta_delta_score`` is TRL's delta_score already multiplied by self.beta.
    per_sequence_loss = -F.logsigmoid(beta_delta_score)
    return per_sequence_loss.mean()


def preference_accuracy(policy_chosen_logp, policy_rejected_logp,
                        reference_chosen_logp, reference_rejected_logp):
    """Fraction of pairs whose policy margin beats the reference margin."""
    logits = preference_logit(
        policy_chosen_logp, policy_rejected_logp,
        reference_chosen_logp, reference_rejected_logp, beta=1.0,
    )
    return (logits > 0).float().mean()
