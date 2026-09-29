"""PPO from scratch, the simple way: the paper's algorithm on the multi-step toy.

How to work through it:
  1. Fill the TODOs in STAGE 1 below (each TODO is one line).
  2. Try it:    python SimplePPO/from_scratch/simple_ppo.py    prints your results next to the expected ones
  3. Check it:  ./scripts/run_simple_ppo.sh check              stops at the first stage that is not right yet
  4. Move on to the next stage.

Where this sits. Surrogates/ ended with: PPO = the loop with L^CLIP in the slot,
plus a value loss, an entropy bonus and GAE. The toy track had one decision per
question; here the agent makes THREE (search or skip, three times), and the
answer's reward only comes at the end. That is the one new idea: with several
steps, each step needs its own advantage, and a critic V(s) helps build it.

    stage 1  the advantage per step     GAE (eq. 11-12)
    stage 2  L^CLIP                     your Surrogates clip_loss, one sample per step
    stage 3  L^VF and S                 the critic's loss, and the entropy bonus
    stage 4  the loop                   K epochs of minibatches on eq. 9
    stage 5  no code                    your PPO on the toy, and what each piece buys

Everything is plain: `(episodes, 3)` tensors, `dist.log_prob`, `.mean()`. No
tokens, no masks -- PPO/ adds those next, for LLMs.

The toy (Policy, Critic, rollout) is in ../env.py. Try not to open
../simple_ppo.py (the reference) -- the hints below are enough.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env import SEARCH, SKIP, Critic, Policy  # noqa: E402,F401


# ============================================================ STAGE 1: the advantage per step, GAE
def compute_gae(rewards, values, gamma, lam):
    """Advantages and returns for (episodes, turns) tensors, by GAE (paper eq. 11-12).

    On the toy track, one decision got one advantage: reward - mean(reward). Now
    an episode has three decisions and its reward is spread over them (a -0.1
    for each search, the answer's quality at the end). Which decision deserves
    the credit? The critic's predictions V(s_t) answer that, one step at a time:

        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)          (eq. 12)
        A_t     = delta_t + gamma * lam * A_{t+1}            (eq. 11, walked backwards)

    delta_t is the critic's SURPRISE at turn t: before acting it expected V(s_t);
    one turn later it has the reward r_t and a new guess V(s_{t+1}). If
    r_t + gamma * V(s_{t+1}) beats V(s_t), the action did better than expected.
    A_t adds up the surprises from t on, each later one shrunk by gamma * lam.

    lam slides between two advantages you know:
        lam = 0:  A_t = delta_t                           one step, trusting the critic
        lam = 1:  A_t = (rewards from t on) - V(s_t)      the outcome minus a baseline --
                                                          your toy advantage, with the critic as baseline

    Why backwards: A_t needs A_{t+1}, so start at the last turn. Past the last
    turn nothing follows, so V = 0 and A = 0 there. Every episode here has the
    same 3 turns, so there is no mask. Each `[:, t]` handles all episodes at once.

    Worked example, one HARD episode: search, search, skip. Each search costs 0.1
    when it happens; two searches make the answer good (1.0). gamma 1, lam 0.8,
    the critic predicting V = [0.5, 0.6, 0.8], so rewards = [-0.1, -0.1, 1.0]:
        t=2: delta =  1.0 + 1 * 0   - 0.8 =  0.20     A_2 = 0.20
        t=1: delta = -0.1 + 1 * 0.8 - 0.6 =  0.10     A_1 = 0.10 + 0.8 * 0.20 = 0.26
        t=0: delta = -0.1 + 1 * 0.6 - 0.5 =  0.00     A_0 = 0.00 + 0.8 * 0.26 = 0.208
        returns = A + V = [0.708, 0.86, 1.0]          (the critic's target, stage 3)
    """
    with torch.no_grad():                               # targets, computed once per batch: no gradient
        nextvalues = 0                                  # V(s_{t+1}); 0 past the last turn
        lastgaelam = 0                                  # A_{t+1};    0 past the last turn
        advantages_reversed = []                        # A_t, collected right to left
        for t in reversed(range(rewards.shape[1])):     # right to left: A_t needs A_{t+1}
            # The critic's surprise at t: reward now, plus the discounted guess one turn later,
            # minus the guess made before acting.
            delta = ...              # TODO stage 1: eq. 12, with nextvalues as V(s_{t+1})
            # This turn's advantage: its own surprise plus the next turn's advantage, shrunk by gamma * lam.
            lastgaelam = ...         # TODO stage 1: eq. 11
            # Moving one turn left: this turn's value becomes "the next value" for turn t-1.
            nextvalues = ...         # TODO stage 1: values at this turn
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)    # back to left-to-right: (episodes, turns)
        # What the critic SHOULD have predicted: its guess plus how much better things turned out.
        returns = ...                # TODO stage 1: the advantage plus the baseline it was measured from
    return advantages, returns


# ============================================================ STAGE 2: L^CLIP (eq. 7)
def policy_loss(policy, states, actions, old_logp, advantages, eps=0.2):
    """-L^CLIP (eq. 7): your Surrogates clip_loss, one sample per step.

    The same four lines as Surrogates stage 4. The only change: each sample
    has a `state` (question type, turn, searches so far) instead of a question
    type -- `policy.dist(states)` looks it up exactly as `policy.dist(qtype)` did.
    Why the change: the right action now depends on the turn and on how many
    searches came before (HARD, turn 2: SEARCH after 1 search, SKIP after 2), so
    the question type alone is no longer the whole state. See "Why `states`,
    not `qtype`" in SimplePPO/README.md.

    Worked example, one step with A = +2 (old p = 0.40):
        ratio 1.1:  min(2.2, 2.2) = 2.2    still rewarded        -> loss -2.2
        ratio 1.5:  min(3.0, 2.4) = 2.4    flat past 1.2: stops  -> loss -2.4
    """
    # Hint: exp(log a - log b) = a / b; `ratio.clamp(lo, hi)`; `torch.min(a, b)`.
    ratio = ...              # TODO stage 2: pi_theta / pi_old for each step's action
    unclipped = ...          # TODO stage 2: ratio * A
    clipped = ...            # TODO stage 2: the ratio squashed into [1 - eps, 1 + eps], times A
    return ...               # TODO stage 2: minus the mean of the smaller of the two


# ============================================================ STAGE 3: L^VF and S (eq. 9)
def value_loss(critic, states, returns):
    """L^VF (eq. 9): 0.5 * (V(s_t) - returns_t)^2, averaged. Trains the critic.

    The critic is a regression: each state's prediction chases the return that
    actually followed it (stage 1). `critic(states)` looks the predictions up.

    Worked example: prediction 0.5, return 0.708 -> 0.5 * (0.5 - 0.708)^2 = 0.0216
    """
    return ...               # TODO stage 3: 0.5 * the mean squared error


def entropy_bonus(policy, states):
    """S (eq. 9): the policy's average entropy over these states. Returned POSITIVE.

    Eq. 9 ADDS c2 * S to the objective, so the policy keeps trying both actions
    before it commits; the loss in stage 4 therefore SUBTRACTS it.

    Worked example: p(search) = 0.4 -> -(0.4 ln 0.4 + 0.6 ln 0.6) = 0.673; p = 1.0 -> 0.
    """
    # Hint: a torch Categorical has `.entropy()`.
    return ...               # TODO stage 3: the mean entropy of policy.dist(states)


# ============================================================ STAGE 4: the loop, Algorithm 1 with eq. 9
def ppo_update(policy, critic, optimizer, batch, epochs=10, minibatch_size=16, eps=0.2,
               vf_coef=0.5, entropy_coeff=0.01):
    """Algorithm 1's inner loop: `epochs` passes over ONE batch, in minibatches, on eq. 9.

    `batch` holds (episodes, turns) tensors from env.rollout, plus the
    advantages and returns from your stage 1. Once every step has its
    advantage, time no longer matters: each step is just one sample, like a
    rollout on the toy track. So the batch is flattened to one list of steps,
    and from there this is Surrogates' loop -- with minibatches, and with all
    three terms of eq. 9 in the slot, negated because optimizers minimise:

        loss = policy_loss + vf_coef * value_loss - entropy_coeff * entropy_bonus

    Returns each loss term averaged over every update.
    """
    # Flatten (episodes, turns) -> (steps,): every step becomes one sample. Given.
    steps = {key: batch[key].reshape(-1) for key in ("states", "actions", "old_logp", "advantages", "returns")}
    n = steps["states"].shape[0]
    totals, updates = {"pg_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0}, 0
    for _ in range(epochs):
        order = torch.randperm(n)                       # a new shuffle every epoch
        for start in range(0, n, minibatch_size):
            # Hint: slice `order` from start, minibatch_size long.
            idx = ...                # TODO stage 4: the indices of this minibatch
            pg = policy_loss(policy, steps["states"][idx], steps["actions"][idx], steps["old_logp"][idx],
                             steps["advantages"][idx], eps)
            vf = value_loss(critic, steps["states"][idx], steps["returns"][idx])
            ent = entropy_bonus(policy, steps["states"][idx])
            loss = ...               # TODO stage 4: eq. 9, negated -- mind the sign of each term

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            updates += 1                                # bookkeeping for the log
            for name, value in (("pg_loss", pg), ("vf_loss", vf), ("entropy", ent)):
                totals[name] += float(value.detach())
    return {name: total / updates for name, total in totals.items()}


# Stage 5 needs no code: check.py trains the toy with your four pieces.


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
        got = [round(x, 4) + 0.0 for x in got.detach().flatten().tolist()]   # + 0.0: no -0.0
        got = got[0] if len(got) == 1 else got
    print(f"  {label:<40} yours: {str(got):<28} expected: {expected}")


def _policy_at(p_search):
    policy = Policy()
    with torch.no_grad():
        policy.logits.copy_(torch.tensor([[1 - p_search, p_search]] * 12).log())
    return policy


if __name__ == "__main__":
    rewards, values = torch.tensor([[-0.1, -0.1, 1.0]]), torch.tensor([[0.5, 0.6, 0.8]])
    one_step = (torch.tensor([0]), torch.tensor([SEARCH]), torch.tensor([0.4]).log(), torch.tensor([2.0]))
    print("Your functions on the worked examples (fill a stage, rerun, compare):\n")
    _show("stage 1  advantages", lambda: compute_gae(rewards, values, 1.0, 0.8)[0], [0.208, 0.26, 0.2])
    _show("stage 1  returns", lambda: compute_gae(rewards, values, 1.0, 0.8)[1], [0.708, 0.86, 1.0])
    _show("stage 2  policy_loss, ratio 1.1", lambda: policy_loss(_policy_at(0.44), *one_step), -2.2)
    _show("stage 2  policy_loss, ratio 1.5", lambda: policy_loss(_policy_at(0.60), *one_step), -2.4)
    critic = Critic()
    with torch.no_grad():
        critic.v.fill_(0.5)
    _show("stage 3  value_loss, 0.5 vs 0.708",
          lambda: value_loss(critic, torch.tensor([0]), torch.tensor([0.708])), 0.0216)
    _show("stage 3  entropy_bonus, p(search) 0.4", lambda: entropy_bonus(Policy(), torch.tensor([0])), 0.673)

    def ten_iterations():
        import env
        curve, _, _ = env.train(sys.modules[__name__], seed=0, iterations=10)
        return f"J 0.388 -> {curve[-1]:.3f}"
    _show("stage 4  10 iterations of PPO", ten_iterations, "J 0.388 -> 0.787")
    print("\nWhen these match, run:  ./scripts/run_simple_ppo.sh check")
