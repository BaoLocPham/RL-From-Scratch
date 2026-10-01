"""DPO, the Rafailov et al. algorithm, on the multi-step toy. Demo: ``python SimpleDPO/run_simple_dpo.py``.

SimpleGRPO dropped PPO's critic. DPO (Direct Preference Optimization,
https://arxiv.org/abs/2305.18290) drops the rest of RL: no reward, no
advantage, no rollouts while training, no ratio to theta_old, no clip. It
trains on pairs that are already labelled -- this attempt is better than that one -- with
a loss that is a logistic regression on the pair:

    for iteration:
        pairs = the next 8 (chosen, rejected) pairs          sampled from pi_ref, before training
        for epoch in range(K):
            for minibatch of M pairs:
                loss = dpo_loss(log pi(chosen), log pi(rejected),     eq. 7
                                log pi_ref(chosen), log pi_ref(rejected))
                zero_grad, backward, step

Three pieces, and nothing else:

    sequence_logp     log pi(attempt): the log-probs of its three turns, summed
    dpo_loss          -log sigmoid(beta * (how much more the policy prefers chosen
                      over rejected than pi_ref does))
    implicit_reward   beta * log(pi / pi_ref): the reward the policy now believes
                      in -- "your language model is secretly a reward model"

pi_ref is not a penalty added to the loss, as in SimpleGRPO: it sits inside
the loss. The pairs and their labels are made in pair_env.py.
"""

import torch
import torch.nn.functional as F


def sequence_logp(policy, states, actions):
    """log pi(attempt) for (pairs, turns) states and actions: the log of the product of its turns' probabilities.

        log pi(a_0, a_1, a_2) = log pi(a_0|s_0) + log pi(a_1|s_1) + log pi(a_2|s_2)

    Summed, not averaged: DPO compares two whole attempts. verl's
    get_batch_logps is this over the response's tokens.
    """
    return policy.dist(states).log_prob(actions).sum(1)


def dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps,
             beta=0.1):
    """DPO's loss (Rafailov et al. eq. 7), averaged over the pairs. verl's compute_online_dpo_loss, "sigmoid".

        logits = [log pi(chosen) - log pi(rejected)] - [log pi_ref(chosen) - log pi_ref(rejected)]
        loss   = -log sigmoid(beta * logits)

    logits > 0: the policy prefers the chosen attempt by more than pi_ref did.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios
    return -F.logsigmoid(beta * logits).mean()              # logsigmoid: log(sigmoid(x)) without underflow


def implicit_reward(policy_logps, reference_logps, beta=0.1):
    """How much more likely the policy has made an attempt than pi_ref did: beta * log(pi(attempt) / pi_ref(attempt)).

    DPO's loss pushes it up for chosen attempts and down for rejected ones, so
    after training it ranks the attempts the way the labels did.
    """
    return beta * (policy_logps - reference_logps)


def dpo_update(policy, optimizer, pairs, epochs=10, minibatch_size=4, beta=0.1):
    """`epochs` passes over one batch of pairs, in minibatches, on dpo_loss.

    Returns the loss, and how often the implicit reward ranks chosen above
    rejected (verl's rewards/accuracies), averaged over every update.
    """
    n = pairs["chosen_states"].shape[0]
    totals, updates = {"loss": 0.0, "accuracy": 0.0}, 0
    for _ in range(epochs):
        order = torch.randperm(n)                           # a new shuffle every epoch
        for start in range(0, n, minibatch_size):
            idx = order[start:start + minibatch_size]
            policy_chosen_logps = sequence_logp(policy, pairs["chosen_states"][idx], pairs["chosen_actions"][idx])
            policy_rejected_logps = sequence_logp(policy, pairs["rejected_states"][idx],
                                                  pairs["rejected_actions"][idx])
            loss = dpo_loss(policy_chosen_logps, policy_rejected_logps, pairs["reference_chosen_logps"][idx],
                            pairs["reference_rejected_logps"][idx], beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            updates += 1                                    # bookkeeping for the log
            chosen = implicit_reward(policy_chosen_logps, pairs["reference_chosen_logps"][idx], beta)
            rejected = implicit_reward(policy_rejected_logps, pairs["reference_rejected_logps"][idx], beta)
            totals["loss"] += float(loss.detach())
            totals["accuracy"] += float((chosen > rejected).float().mean())
    return {name: total / max(updates, 1) for name, total in totals.items()}
