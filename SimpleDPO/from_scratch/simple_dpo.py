"""DPO from scratch, the simple way: Rafailov et al.'s algorithm on the multi-step toy.

How to work through it:
  1. Fill the TODOs in STAGE 1 below (each TODO is one line).
  2. Try it:    python SimpleDPO/from_scratch/simple_dpo.py    prints your results next to the expected ones
  3. Check it:  ./scripts/run_simple_dpo.sh check              stops at the first stage that is not right yet
  4. Move on to the next stage.

Where this sits. SimpleGRPO dropped PPO's critic. DPO (Direct Preference
Optimization, https://arxiv.org/abs/2305.18290) drops the rest of RL: no
reward, no advantage, no rollouts while training, no ratio to theta_old, no
clip. It trains on pairs that are already labelled -- this attempt is better
than that one -- with a loss that is a logistic regression on each pair:

    stage 1  sequence_logp       log pi(attempt): its three turns' log-probs, summed
    stage 2  dpo_loss            -log sigmoid(beta * logits), DPO eq. 7
             implicit_reward     beta * log(pi / pi_ref): the reward the policy believes in
    stage 3  the loop            epochs of minibatches of pairs, on your dpo_loss
    stage 4  no code             your DPO on the toy

Nothing is imported from your earlier exercises: DPO keeps none of PPO's loss.

The toy is SimplePPO's agent, two attempts per question, labelled by which one
is better: see ../pair_env.py. Try not to open ../simple_dpo.py (the reference).
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pair_env import HARD, Policy, all_attempts  # noqa: E402,F401


# ============================================================ STAGE 1: log pi(attempt)
def sequence_logp(policy, states, actions):
    """log pi(attempt) for (pairs, turns) states and actions: the log of the product of its turns' probabilities.

    DPO compares two whole attempts, so it needs one number per attempt. An
    attempt is three choices, each made in its own state, so its probability is
    their product -- in logs, a sum:

        log pi(a_0, a_1, a_2) = log pi(a_0|s_0) + log pi(a_1|s_1) + log pi(a_2|s_2)

    Summed, NOT averaged. verl's get_batch_logps is this over a response's tokens.

    Shapes, for a minibatch of M pairs (M = 4 in dpo_update), one side of each pair:

        states       (M, 3)  long, state ids 0..11 (see env.state_id), one column per turn
        actions      (M, 3)  long, 0 = SKIP, 1 = SEARCH
        policy.dist(states)                     an (M, 3) batch of Categoricals over the 2 actions
        policy.dist(states).log_prob(actions)   (M, 3)  log pi(a_t | s_t), turn by turn
        returns      (M,)    one log-probability per attempt; it must carry gradient

    Worked example, pi_ref (p(search) = 0.4 everywhere), on one HARD question:
        A = (search, search, skip): log 0.4 + log 0.4 + log 0.6 = -0.916 - 0.916 - 0.511 = -2.343
        B = (skip, skip, skip):     log 0.6 * 3                                        = -1.532
    """
    # (M, 3) states and actions -> (M, 3): one log-prob per turn.
    per_turn = ...                  # TODO stage 1: log pi(a_t | s_t) for every turn, shape (pairs, turns)
    # (M, 3) -> (M,): add up dim 1, the turns.
    return ...                      # TODO stage 1: summed over the turns, shape (pairs,)


# ============================================================ STAGE 2: the loss (DPO eq. 7)
def dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps,
             beta=0.1):
    """DPO's loss, averaged over the pairs. verl's compute_online_dpo_loss, loss_type="sigmoid".

    The idea: make the chosen attempt more likely and the rejected one less
    likely -- measured against pi_ref, so the policy is judged by how far it has
    MOVED from where it started:

        pi_logratios  = log pi(chosen)     - log pi(rejected)       how much the policy prefers chosen
        ref_logratios = log pi_ref(chosen) - log pi_ref(rejected)   how much pi_ref did
        logits        = pi_logratios - ref_logratios                > 0: the policy moved toward chosen
        loss          = mean over pairs of -log sigmoid(beta * logits)

    -log sigmoid(x) is the loss of a yes/no classifier whose answer should be
    "yes, chosen is better": large when x < 0, near 0 when x is large. (Why
    this exact form: the walkthrough, step 4.)

    Use F.logsigmoid, not torch.log(torch.sigmoid(...)): when the policy gets a
    pair badly wrong, sigmoid underflows to 0 and its log is -inf. The names are
    verl's.

    Shapes: every input is (M,), one number per pair; the policy ones carry
    gradient, the reference ones do not (they were computed once, from pi_ref).
    Returns ()  a scalar.

    Worked example, the pair above, beta 0.1:
        pi = pi_ref:                    logits 0.0             loss -log sigmoid(0)     = 0.693
        pi searching with p = 0.6:      log pi(A) = -1.938, log pi(B) = -2.749
                                        pi_logratios 0.811, ref_logratios -0.811, logits 1.622
                                        loss -log sigmoid(0.1622) = 0.615
    """
    # (M,) - (M,) -> (M,) for each of the three lines.
    pi_logratios = ...              # TODO stage 2: log pi(chosen) - log pi(rejected)
    ref_logratios = ...             # TODO stage 2: the same under pi_ref
    logits = ...                    # TODO stage 2: how much more the policy prefers chosen than pi_ref did
    # (M,) -> (): the negative log-likelihood, averaged.
    return ...                      # TODO stage 2: the mean of -logsigmoid(beta * logits)


def implicit_reward(policy_logps, reference_logps, beta=0.1):
    """How much more likely the policy has made an attempt than pi_ref did: beta * log(pi(attempt) / pi_ref(attempt)).

    DPO's loss pushes it up for chosen attempts and down for rejected ones, so
    after training it ranks the attempts the way the labels did: the policy has
    become a scorer of attempts. Only differences between attempts at the same
    question mean anything. verl logs it as rewards_chosen / rewards_rejected.

    Shapes: (M,) and (M,) -> (M,).

    Worked example: log pi = -1.177, log pi_ref = -2.343, beta 0.1: 0.1 * 1.166 = 0.117
    """
    return ...                      # TODO stage 2: beta times the log-ratio to pi_ref


# ============================================================ STAGE 3: the loop
def dpo_update(policy, optimizer, pairs, epochs=10, minibatch_size=4, beta=0.1):
    """`epochs` passes over one batch of pairs, in minibatches, on your dpo_loss.

    SimpleGRPO's grpo_update, with the pairs in place of the steps: no
    advantage, no old log-probs, no clip. The loss does not depend on which
    policy sampled the pairs, so reusing them for several epochs needs no
    ratio to protect it.

    Shapes, for 8 pairs (one per question):

        pairs["chosen_states"], pairs["chosen_actions"]         (8, 3)  long: the preferred attempt
        pairs["rejected_states"], pairs["rejected_actions"]     (8, 3)  long: the other one
        pairs["reference_chosen_logps"], [..."rejected_logps"]  (8,)    log pi_ref of each, fixed
        idx                                                     (4,)    positions into the 8 pairs
        pairs[key][idx]                                         (4, 3) or (4,): this minibatch
        policy_chosen_logps, policy_rejected_logps              (4,)    your stage 1, with gradient
        loss                                                    ()

    Returns the loss, and how often the implicit reward ranks chosen above
    rejected, averaged over every update.
    """
    n = pairs["chosen_states"].shape[0]
    totals, updates = {"loss": 0.0, "accuracy": 0.0}, 0
    for _ in range(epochs):
        order = torch.randperm(n)                       # a new shuffle every epoch
        for start in range(0, n, minibatch_size):
            idx = order[start:start + minibatch_size]
            # pairs["chosen_states"][idx], pairs["chosen_actions"][idx]: (4, 3) each -> (4,)
            policy_chosen_logps = ...                   # TODO stage 3: your stage 1 on this minibatch's chosen attempts
            policy_rejected_logps = ...                 # TODO stage 3: and on its rejected attempts
            # four (4,) tensors -> (): the reference log-probs come from pairs, at [idx]
            loss = ...                                  # TODO stage 3: your stage 2 loss, with beta

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            updates += 1                                # bookkeeping for the log
            chosen = implicit_reward(policy_chosen_logps, pairs["reference_chosen_logps"][idx], beta)
            rejected = implicit_reward(policy_rejected_logps, pairs["reference_rejected_logps"][idx], beta)
            totals["loss"] += float(loss.detach())
            totals["accuracy"] += float((chosen > rejected).float().mean())
    return {name: total / max(updates, 1) for name, total in totals.items()}


# Stage 4 needs no code: check.py trains the toy with your three pieces.


# ============================================================ playground
def _show(label, fn, expected):
    """Run one of your functions and print it next to the expected value."""
    try:
        got = fn()
    except Exception as exc:                           # unfinished TODOs land here
        got = f"not done yet ({type(exc).__name__})"
    if got is Ellipsis:
        got = "not done yet"
    elif isinstance(got, torch.Tensor):
        got = [round(x, 3) + 0.0 for x in got.detach().flatten().tolist()]   # + 0.0: no -0.0
        got = got[0] if len(got) == 1 else got
    print(f"  {label:<38} yours: {str(got):<28} expected: {expected}")


def _policy_at(p_search):
    policy = Policy()
    with torch.no_grad():
        policy.logits.copy_(torch.tensor([[1 - p_search, p_search]] * 12).log())
    return policy


if __name__ == "__main__":
    states, actions, _ = all_attempts(HARD)
    pair = (states[[6, 0]], actions[[6, 0]])           # A = (search, search, skip), B = (skip, skip, skip)

    def loss_at(p_search):
        mine, ref = sequence_logp(_policy_at(p_search), *pair), sequence_logp(Policy(), *pair)
        return dpo_loss(mine[:1], mine[1:], ref[:1], ref[1:], 0.1)

    print("Your functions on the worked examples (fill a stage, rerun, compare):\n")
    _show("stage 1  log pi_ref of A and B", lambda: sequence_logp(Policy(), *pair), [-2.343, -1.532])
    _show("stage 2  dpo_loss at pi = pi_ref", lambda: loss_at(0.4), 0.693)
    _show("stage 2  dpo_loss at p(search) 0.6", lambda: loss_at(0.6), 0.615)
    _show("stage 2  implicit_reward, example",
          lambda: implicit_reward(torch.tensor([-1.177]), torch.tensor([-2.343]), 0.1), 0.117)

    def ten_iterations():
        import pair_env
        curve, _ = pair_env.train(sys.modules[__name__], seed=0, iterations=10)
        return f"J 0.388 -> {curve[-1]:.3f}"
    _show("stage 3  10 iterations of DPO", ten_iterations, "J 0.388 -> 0.871")
    print("\nWhen these match, run:  ./scripts/run_simple_dpo.sh check")
