"""SimplePPO's three-turn toy, sampled in groups and graded right or wrong. Given code.

The agent is SimplePPO's, unchanged: three turns of SEARCH (costs 0.1) or SKIP,
then it answers; the same 12 states, the same 12 x 2 table of logits (see
SimplePPO/env.py). Two things change, both to make it GRPO's setting:

1. Groups. GRPO samples several answers to the SAME question and compares them
   with each other. A batch is `questions` questions, each answered
   `group_size` times, stored group by group:

       episodes 0..G-1     question 0, G attempts
       episodes G..2G-1    question 1, G attempts        ...

2. A verifiable reward. An LLM trained with GRPO is usually graded right or
   wrong by a checker. So here the answer is CORRECT (+1) or WRONG (-1), with
   the chance of being right set by the question and the searches:

                  0 searches  1 search  2 searches  3 searches
       HARD           0%         50%       100%        100%
       EASY          100%        95%        90%         85%

   Its average, 2 * p(correct) - 1, is exactly SimplePPO's answer quality
   (HARD -1, 0, 1, 1; EASY 1, 0.9, 0.8, 0.7). So the true J, the best policy
   (search twice on HARD, never on EASY) and BEST_J = 0.9 are all the same as
   SimplePPO's; only the noise is different: a coin instead of a bell curve.

The reference policy pi_ref, which GRPO's KL term holds the policy near, is
the starting policy, frozen.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SimplePPO"))
from env import (ANSWER_QUALITY, BEST_J, EASY, HARD, N_STATES, SEARCH, SEARCH_COST, SKIP,  # noqa: E402,F401
                 TURNS, Critic, Policy, describe, state_id, true_reward)

P_CORRECT = (1 + ANSWER_QUALITY) / 2        # [question type][searches so far]: the chance the answer is right


@torch.no_grad()
def rollout(policy, ref_policy, questions, group_size, critic=None):
    """`questions` questions, each attempted `group_size` times by the policy as it is NOW (theta_old).

    Every tensor has one row per episode, (questions * group_size, 3), group by group:

    rewards   -0.1 on each turn that searched; +1 (correct) or -1 (wrong) added on the last turn
    old_logp  log pi_theta_old(a_t | s_t), frozen
    ref_logp  log pi_ref(a_t | s_t), the reference policy's log-prob of the same action
    correct   (episodes,) 1.0 if the answer was right
    values    V(s_t), only if a critic is given (for comparing with PPO)
    """
    qtype = torch.randint(0, 2, (questions,)).repeat_interleave(group_size)    # one type per question, G attempts
    n = qtype.shape[0]
    searches = torch.zeros(n, dtype=torch.long)
    states, actions, old_logp, ref_logp, values = [], [], [], [], []
    rewards = torch.zeros(n, TURNS)
    for turn in range(TURNS):
        state = state_id(qtype, turn, searches)
        action = policy.dist(state).sample()
        states.append(state)
        actions.append(action)
        old_logp.append(policy.dist(state).log_prob(action))
        ref_logp.append(ref_policy.dist(state).log_prob(action))
        if critic is not None:
            values.append(critic(state))
        rewards[:, turn] -= SEARCH_COST * action              # the search is paid for now
        searches = searches + action
    correct = (torch.rand(n) < P_CORRECT[qtype, searches]).float()
    rewards[:, -1] += 2 * correct - 1                         # +1 right, -1 wrong
    batch = {"states": torch.stack(states, 1), "actions": torch.stack(actions, 1),
             "old_logp": torch.stack(old_logp, 1), "ref_logp": torch.stack(ref_logp, 1),
             "rewards": rewards, "correct": correct, "qtype": qtype, "searches": searches}
    if critic is not None:
        batch["values"] = torch.stack(values, 1)
    return batch


def dead_groups(rewards, group_size):
    """The fraction of groups whose attempts all scored the same: their std is 0 and they teach nothing."""
    scores = rewards.sum(1).view(-1, group_size)
    return float((scores.std(1) < 1e-6).float().mean())


def train(impl, seed, iterations=60, questions=2, group_size=8, lr=0.3, advantage=None, **update):
    """The GRPO loop: collect groups -> group advantages -> grpo_update, `iterations` times.

    `impl` is a module with group_advantage and grpo_update: the reference
    (SimpleGRPO/simple_grpo.py) or yours. `advantage` replaces the group
    advantage for the comparisons in run_simple_grpo.py. Returns the true J
    after every iteration, the trained policy, and the fraction of dead groups
    in every batch.
    """
    torch.manual_seed(seed)
    policy, ref_policy = Policy(), Policy()                   # pi_ref: the starting policy, never trained
    optimizer = torch.optim.SGD(policy.parameters(), lr=lr)
    curve, dead = [], []
    for _ in range(iterations):
        batch = rollout(policy, ref_policy, questions, group_size)             # theta_old
        if advantage is None:
            batch["advantages"] = impl.group_advantage(batch["rewards"], group_size)
        else:
            batch["advantages"] = advantage(batch)
        dead.append(dead_groups(batch["rewards"], group_size))
        impl.grpo_update(policy, optimizer, batch, **update)                  # K epochs of minibatches
        curve.append(true_reward(policy))
    return curve, policy, dead
