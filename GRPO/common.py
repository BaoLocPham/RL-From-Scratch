"""GRPO as verl implements it.

Follows `verl/trainer/ppo/core_algos.py`. verl keeps every advantage estimator
in that one file, so the pieces GRPO shares with PPO -- ``agg_loss``,
``compute_policy_loss``, ``kl_penalty``, the masked helpers -- are imported
from ``PPO/common.py`` rather than copied. What is actually GRPO's own is
short: one advantage function.

The idea in a sentence: PPO learns a critic to predict expected reward and
subtracts it; GRPO samples several responses to the *same* prompt and
subtracts their own mean. No value head, no ``compute_gae_advantage_return``,
no ``compute_value_loss`` -- which is most of PPO's cost and most of its
tuning surface.
"""

import importlib.util
from pathlib import Path

import torch


def _load(name, path):
    """Import a module by path. Every directory here has a file named common.py."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ppo = _load("ppo_common", Path(__file__).resolve().parent.parent / "PPO" / "common.py")

# Re-exported unchanged: in verl these live beside the GRPO advantage in one
# file, and a GRPO trainer calls exactly these.
agg_loss = _ppo.agg_loss
compute_policy_loss = _ppo.compute_policy_loss
kl_penalty = _ppo.kl_penalty
masked_mean = _ppo.masked_mean
masked_whiten = _ppo.masked_whiten


def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index,
                                   epsilon=1e-6, norm_adv_by_std_in_grpo=True,
                                   config=None):
    """Return ``(advantages, returns)`` standardized within each prompt group.

    Args:
        token_level_rewards: ``(bs, response_length)``. Outcome supervision, so
            each row carries one scalar reward; this function begins by summing
            the row back down to that scalar.
        response_mask: ``(bs, response_length)``.
        index: length-``bs`` array of prompt ids. Rows sharing an id were
            sampled from the same prompt and form one group. In verl this is
            ``non_tensor_batch["uid"]``.
        epsilon: added to the group std before dividing.
        norm_adv_by_std_in_grpo: ``True`` is GRPO. ``False`` subtracts the mean
            without dividing, which is **Dr.GRPO** -- the single flag that paper
            changes. Dividing by a per-group std makes the update size depend on
            how much that group happened to disagree, which correlates with
            response length and shows up as a length bias.

    Both returned tensors are the same object: with outcome supervision there is
    nothing for a critic to regress against, so ``returns`` is meaningless here.
    verl returns the pair anyway to keep one signature across every estimator.

    Two details worth copying exactly:

    * A group of size one gets ``mean=0, std=1`` rather than its own mean, which
      would make its advantage identically zero and waste the sample.
    * The std is ``torch.std``'s default sample (n-1) std, not the population
      std. Write-ups of GRPO frequently use the population std and get visibly
      different numbers on small groups.
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        id2score = {}
        for i in range(scores.shape[0]):
            id2score.setdefault(index[i], []).append(scores[i])

        id2mean, id2std = {}, {}
        for idx, group in id2score.items():
            if len(group) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(group) > 1:
                id2mean[idx] = torch.mean(torch.tensor(group))
                id2std[idx] = torch.std(torch.tensor(group))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(scores.shape[0]):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        # One scalar per response, broadcast across every token it owns. Each
        # token of a response carries the whole response's verdict -- there is
        # no per-token credit assignment in outcome-supervised GRPO.
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores
