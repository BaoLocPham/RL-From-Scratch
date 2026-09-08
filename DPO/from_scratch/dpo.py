"""DPO from scratch. Fill each TODO, then run ``check.py``."""

import torch
import torch.nn.functional as F


# ============================================================ STAGE 1
def sequence_log_probs(logits, token_ids, mask):
    """Return one summed completion log-probability per batch row."""
    # TODO stage 1: select target-token likelihoods and exclude padding.
    pass


# ============================================================ STAGE 2
def preference_logit(policy_chosen_logp, policy_rejected_logp,
                     reference_chosen_logp, reference_rejected_logp, beta=0.1):
    """Return the reference-corrected preference logit for every pair."""
    # TODO stage 2: compare TRL's chosen_logratios and rejected_logratios.
    pass


# ============================================================ STAGE 3
def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             reference_chosen_logp, reference_rejected_logp, beta=0.1):
    """Return the mean binary preference loss."""
    # TODO stage 3: turn the stage-2 logit into a stable scalar objective.
    pass


# ============================================================ STAGE 4
def preference_accuracy(policy_chosen_logp, policy_rejected_logp,
                        reference_chosen_logp, reference_rejected_logp):
    """Return the share of policy pair margins that improve on the reference."""
    # TODO stage 4: report a thresholded diagnostic, not a training loss.
    pass
