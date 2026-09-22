"""verl's GRPO from scratch. Fill the TODOs, then run ``check.py``.

GRPO's own contribution is one function. Everything else it needs --
``agg_loss``, ``compute_policy_loss``, ``kl_penalty`` -- is PPO's, and in verl
lives in the same ``core_algos.py``. Build those in ``PPO/from_scratch/`` first;
this exercise imports your versions, so ``PPO/from_scratch/check.py`` has to
pass before this one will.

The idea: PPO trains a critic to predict expected reward and subtracts it.
GRPO samples several responses to the same prompt and subtracts their own mean.
No value head, so no GAE and no value loss -- most of PPO's cost and most of
its tuning surface, gone.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "PPO" / "from_scratch"))
from ppo import agg_loss, compute_policy_loss, kl_penalty, masked_mean  # noqa: F401,E402


# ============================================================ STAGE 1
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index,
                                   epsilon=1e-6, norm_adv_by_std_in_grpo=True,
                                   config=None):
    """Return ``(advantages, returns)`` standardized within each prompt group.

    Args:
        token_level_rewards: ``(bs, response_length)``. Outcome supervision, so
            each row carries a single scalar reward spread across it. Step one
            is summing the row back down to that scalar.
        response_mask: ``(bs, response_length)``.
        index: length-``bs`` array of prompt ids. Rows sharing an id came from
            the same prompt and form one group. verl passes
            ``non_tensor_batch["uid"]`` here.
        norm_adv_by_std_in_grpo: ``True`` is GRPO. ``False`` subtracts the mean
            and does NOT divide -- that is **Dr.GRPO**, the single flag that
            paper changes. Dividing by a per-group std ties the update size to
            how much that group happened to disagree, which correlates with
            response length and shows up as a length bias.

    Three details that are easy to get wrong:

    * A group of size one takes ``mean=0, std=1``, NOT its own mean. Its own
      mean would make the advantage identically zero and waste the sample.
    * ``torch.std`` defaults to the sample (n-1) std. Write-ups of GRPO often
      use the population std and get different numbers on small groups. Match
      the code.
    * Return the same tensor twice. With outcome supervision there is no critic
      to regress, so ``returns`` is meaningless -- verl returns the pair anyway
      so every estimator shares one signature.
    """
    scores = ...                     # TODO stage 1: one scalar reward per response

    with torch.no_grad():
        id2score = {}
        for i in range(scores.shape[0]):
            ...                      # TODO stage 1: bucket scores[i] under its prompt id

        id2mean, id2std = {}, {}
        for idx, group in id2score.items():
            if len(group) == 1:
                id2mean[idx] = ...   # TODO stage 1: the singleton special case
                id2std[idx] = ...    # TODO stage 1
            elif len(group) > 1:
                id2mean[idx] = ...   # TODO stage 1
                id2std[idx] = ...    # TODO stage 1: sample std, not population
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(scores.shape[0]):
            if norm_adv_by_std_in_grpo:
                scores[i] = ...      # TODO stage 1: centre AND scale, guarded by epsilon
            else:
                scores[i] = ...      # TODO stage 1: Dr.GRPO -- centre only

        scores = ...                 # TODO stage 1: broadcast each scalar across its
                                     # response's real tokens; every token of a response
                                     # carries that whole response's verdict

    return scores, scores
