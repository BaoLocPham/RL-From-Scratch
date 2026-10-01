"""SimpleGRPO's toy, two attempts per question, and a rater who says which one is better. Given code.

The agent is SimplePPO's, unchanged: three turns of SEARCH (costs 0.1) or SKIP,
then it answers; the same 12 states, the same 12 x 2 table of logits (see
SimplePPO/env.py). What changes is what training gets to see.

1. Pairs. Each question is attempted twice -- a group of 2, in SimpleGRPO's
   terms -- stored pair by pair:

       episodes 0, 1    question 0
       episodes 2, 3    question 1        ...

2. A preference, not a reward. DPO never sees a number. A rater looks at the
   two attempts and says which one is better, and that is all. The rater is
   the DPO paper's model of a human (Bradley-Terry, eq. 1):

       p(attempt 1 preferred over attempt 2) = sigmoid(r(attempt 1) - r(attempt 2))

   where r is the attempt's quality: the average total reward of its number of
   searches, SimplePPO's answer quality minus 0.1 per search:

                  0 searches  1 search  2 searches  3 searches
       HARD          -1.0       -0.1       0.8         0.7
       EASY           1.0        0.8       0.6         0.4

   The rater judges the answer, not the luck: on a HARD question one search is
   a coin flip, and the rater knows it. The verdicts are still noisy: on HARD,
   two searches beat none only 86% of the time (sigmoid(1.8)), and two beat
   three only 52% of the time (sigmoid(0.1)).

   `labels="outcome"` ranks the two by the rewards they actually got instead
   (+1 right, -1 wrong, -0.1 per search, as in SimpleGRPO), and drops ties: the
   way verl's online-DPO recipe ranks responses with a checker.

The reference policy pi_ref is the starting policy, frozen, as in SimpleGRPO.
DPO's pi_ref is also where its data comes from, if it is offline: the
published DPO trains on pairs sampled once, from the model before training.
"""

import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SimpleGRPO"))
from group_env import (ANSWER_QUALITY, BEST_J, EASY, HARD, SEARCH, SEARCH_COST, SKIP, TURNS,  # noqa: E402,F401
                       Policy, describe, rollout, state_id, true_reward)

# QUALITY[question type][searches]: the average total reward of an attempt. The rater's r, and J's.
QUALITY = ANSWER_QUALITY - SEARCH_COST * torch.arange(TURNS + 1)


@torch.no_grad()
def collect_pairs(policy, ref_policy, questions, sequence_logp, labels="rater"):
    """`questions` questions, each attempted twice by `policy`, turned into (chosen, rejected) pairs.

    Every tensor has one row per pair. With the rater there is one pair per
    question; with labels="outcome", tied pairs are dropped.

    chosen_states, chosen_actions         (pairs, 3)  the preferred attempt, turn by turn
    rejected_states, rejected_actions     (pairs, 3)  the other one
    reference_chosen_logps                (pairs,)    log pi_ref(chosen attempt), by `sequence_logp`
    reference_rejected_logps              (pairs,)    log pi_ref(rejected attempt)
    qtype, chosen_searches, rejected_searches   (pairs,)  for printing

    `sequence_logp` is passed in so the reference log-probs come from the same
    code as the policy's, as in verl, where the reference model runs the
    actor's own forward pass.
    """
    batch = rollout(policy, ref_policy, questions, 2)          # episodes 2i and 2i + 1: question i
    firsts = torch.arange(questions) * 2
    if labels == "rater":
        quality = QUALITY[batch["qtype"], batch["searches"]].view(-1, 2)          # (questions, 2)
        first_wins = torch.rand(questions) < torch.sigmoid(quality[:, 0] - quality[:, 1])
        keep = torch.ones(questions, dtype=torch.bool)
    elif labels == "outcome":
        scores = batch["rewards"].sum(1).view(-1, 2)                             # (questions, 2)
        first_wins = scores[:, 0] > scores[:, 1]
        keep = (scores[:, 0] - scores[:, 1]).abs() > 1e-6                        # a tie says nothing
    else:
        raise ValueError(f"unknown labels: {labels}")
    chosen = (firsts + (~first_wins).long())[keep]
    rejected = (firsts + first_wins.long())[keep]

    pairs = {"chosen_states": batch["states"][chosen], "chosen_actions": batch["actions"][chosen],
             "rejected_states": batch["states"][rejected], "rejected_actions": batch["actions"][rejected],
             "qtype": batch["qtype"][chosen], "chosen_searches": batch["searches"][chosen],
             "rejected_searches": batch["searches"][rejected]}
    if chosen.numel() == 0:                                   # every pair tied
        pairs["reference_chosen_logps"] = pairs["reference_rejected_logps"] = torch.zeros(0)
    else:
        pairs["reference_chosen_logps"] = sequence_logp(ref_policy, pairs["chosen_states"], pairs["chosen_actions"])
        pairs["reference_rejected_logps"] = sequence_logp(ref_policy, pairs["rejected_states"],
                                                          pairs["rejected_actions"])
    return pairs


def train(impl, seed, iterations=60, questions=8, lr=3.0, online=False, labels="rater", **update):
    """DPO's loop: 8 pairs -> dpo_update (K epochs of minibatches), `iterations` times.

    online=False is the published DPO: every pair is sampled up front, from
    pi_ref, and iteration i trains on the i-th 8 of them. online=True samples
    each iteration's pairs from the policy as it is now, as verl's online-DPO
    recipe does. Either way, 16 attempts per iteration, SimpleGRPO's budget.

    lr 3.0 is SimpleGRPO's 0.3 divided by beta = 0.1: DPO's gradient carries a
    factor of beta (see the walkthrough, step 6), so the same SGD needs a step
    1 / beta times bigger. `impl` is a module with sequence_logp and
    dpo_update: the reference (SimpleDPO/simple_dpo.py) or yours. Returns the
    true J after every iteration, and the trained policy.
    """
    torch.manual_seed(seed)
    policy, ref_policy = Policy(), Policy()                   # pi_ref: the starting policy, never trained
    optimizer = torch.optim.SGD(policy.parameters(), lr=lr)
    if not online:                                            # the whole dataset, from pi_ref, before training
        dataset = [collect_pairs(ref_policy, ref_policy, questions, impl.sequence_logp, labels)
                   for _ in range(iterations)]
    curve = []
    for iteration in range(iterations):
        if online:
            pairs = collect_pairs(policy, ref_policy, questions, impl.sequence_logp, labels)
        else:
            pairs = dataset[iteration]
        impl.dpo_update(policy, optimizer, pairs, **update)
        curve.append(true_reward(policy))
    return curve, policy


# ------------------------------------------------------------ the exact answers
# What DPO is aiming at, computed by enumerating all 8 attempts at each question type.

def all_attempts(qtype):
    """Every action sequence for one question type, as (states, actions, searches): (8, 3), (8, 3), (8,)."""
    states, actions, searches = [], [], []
    for sequence in itertools.product((SKIP, SEARCH), repeat=TURNS):
        count, row = 0, []
        for turn, action in enumerate(sequence):
            row.append(state_id(qtype, turn, count))
            count += action
        states.append(row)
        actions.append(list(sequence))
        searches.append(count)
    return torch.tensor(states), torch.tensor(actions), torch.tensor(searches)


def optimal_policy(beta):
    """pi*_beta, the policy DPO's loss is minimised by: pi_ref(attempt) * exp(r / beta), normalised.

    It is exactly representable by the 12 x 2 table: its p(search) in each state
    is the weight of the attempts that search there over the weight of all the
    attempts that reach that state.
    """
    reference = Policy()
    policy = Policy()
    with torch.no_grad():
        for qtype in (HARD, EASY):
            states, actions, searches = all_attempts(qtype)
            logp_ref = reference.dist(states).log_prob(actions).sum(1)
            weight = torch.exp(logp_ref + QUALITY[qtype, searches] / beta)      # unnormalised pi*(attempt)
            for turn in range(TURNS):
                done = actions[:, :turn].sum(1)                                  # searches before this turn
                for count in range(turn + 1):
                    here = done == count
                    p = float(weight[here & (actions[:, turn] == SEARCH)].sum() / weight[here].sum())
                    policy.logits[state_id(qtype, turn, count)] = torch.tensor([1 - p, p]).log()
    return policy


@torch.no_grad()
def implicit_rewards(impl, policy, beta):
    """The reward the policy implies, beta * log(pi / pi_ref), per question type and number of searches.

    Averaged over the attempts with the same number of searches, then shifted
    so that 0 searches is 0: a reward is only ever defined up to a constant per
    question. Returns a (2, 4) tensor to compare with QUALITY - QUALITY[:, :1].
    """
    reference = Policy()
    out = torch.zeros(2, TURNS + 1)
    for qtype in (HARD, EASY):
        states, actions, searches = all_attempts(qtype)
        reward = impl.implicit_reward(impl.sequence_logp(policy, states, actions),
                                      impl.sequence_logp(reference, states, actions), beta)
        for k in range(TURNS + 1):
            out[qtype, k] = reward[searches == k].mean()
    return out - out[:, :1]
