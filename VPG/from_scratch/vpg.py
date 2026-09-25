"""Vanilla policy gradient from scratch.

How to work through it:
  1. Fill the TODOs in STAGE 1 below (each TODO is one line).
  2. Try it:    python VPG/from_scratch/vpg.py       prints your results next to the expected ones
  3. Check it:  python VPG/from_scratch/check.py     stops at the first stage that is not right yet
  4. Move on to the next stage.

Every stage uses the same running example as the notes: 10 rollouts on HARD
questions, collected at p(tool) = 0.40. 4 called the tool and went well
(advantage +3); 6 answered directly and went badly (advantage -2).

Try not to open ../vpg.py (the reference) -- the hints below are enough.

    J(theta) = E_{a ~ pi_theta}[ R(a) ]                  the real goal (training never sees it)
    L^PG     = mean over the batch of log pi(a|s) * A    (eq. 2) its stand-in, from one batch
"""

import torch
import torch.nn as nn


# ============================================================ STAGE 1: the goal, J
def expected_reward(probs):
    """J(theta): the average reward this policy earns, computed exactly.

    ``probs[type][action]`` is the chance of each action on each question type,
    and MEAN_REWARD[type][action] (at the bottom of this file) is its average
    reward. For each type, weight each reward by its chance and add them up;
    then average the two types, since HARD and EASY are 50/50.

    Worked example, the starting policy probs = [[0.6, 0.4], [0.6, 0.4]]:
        HARD:  0.6 * 0.2 + 0.4 * 1.0 = 0.52
        EASY:  0.6 * 0.8 + 0.4 * 0.5 = 0.68
        J    = (0.52 + 0.68) / 2     = 0.60
    """
    # Hint: `probs * MEAN_REWARD` multiplies element by element (same [2, 2] shape);
    #       `.sum(dim=-1)` adds along the last axis, the actions.
    per_type = ...           # TODO stage 1: [HARD's expected reward, EASY's expected reward] -> [0.52, 0.68]
    return ...               # TODO stage 1: the average of those two numbers -> 0.60


# ============================================================ STAGE 2: the advantage
def compute_advantage(reward):
    """A_t: was this rollout better (+) or worse (-) than the batch's average?

    Worked example, rewards [1, 2, 3, 6]: the average is 3, so
        advantages = [1-3, 2-3, 3-3, 6-3] = [-2, -1, 0, 3]
    Just subtract the average -- do not also divide by the standard deviation.
    """
    # Hint: `reward.mean()` is the batch average.
    return ...               # TODO stage 2: reward minus its average -> [-2, -1, 0, 3]


# ============================================================ STAGE 3: the loss, -L^PG
def pg_loss(policy, qtype, action, advantage, old_logp):
    """-L^PG (eq. 2): minus the average of log pi(a|s) * A over the batch.

    Step 1: log pi(a|s) -- how likely the CURRENT policy finds each action taken.
    Step 2: multiply by the advantage, average over the batch, and put a minus
            sign in front (PyTorch optimizers MINIMISE, and we want L^PG to go UP).

    Worked example, the running example at p(tool) = 0.40:
        tool calls:     log(0.40) = -0.916,  times A = +3  ->  -2.749   (x4)
        direct answers: log(0.60) = -0.511,  times A = -2  ->  +1.022   (x6)
        L^PG  = (4 * -2.749 + 6 * 1.022) / 10 = -0.487
        loss  = -L^PG                          = +0.487

    ``old_logp`` is passed in, but do NOT use it: L^PG has no memory of the model
    that collected the batch. That is exactly vanilla PG's flaw (stage 5).
    """
    # Hint: `policy.dist(qtype)` gives each question's action distribution;
    #       its `.log_prob(action)` gives log pi of the action that was taken.
    logp = ...               # TODO stage 3, step 1: one log-probability per rollout -> shape [10]
    return ...               # TODO stage 3, step 2: minus the mean of logp * advantage -> 0.487


# ============================================================ STAGE 4: one update
def vpg_update(policy, optimizer, batch, updates=1):
    """Change the policy using one batch: compute the loss, get its gradient, step.

    Vanilla PG does this ONCE per batch (updates=1). The loop is there so stage 5
    can show what happens if you keep updating on the same batch.
    """
    for _ in range(updates):
        # Hint: the standard PyTorch three lines, in this order:
        #       optimizer.zero_grad()  ->  loss.backward()  ->  optimizer.step()
        loss = ...           # TODO stage 4: your pg_loss on this batch  (pg_loss(policy, *batch))
        ...                  # TODO stage 4: clear the gradient left over from the previous update
        ...                  # TODO stage 4: compute the gradient of the loss
        ...                  # TODO stage 4: take the update -- the only line that changes the model


# Stage 5 needs no code: check.py runs your four functions on an unlucky batch
# and shows what happens when one batch is reused for 100 updates.


# ============================================================ the toy (given -- no need to change)
HARD, EASY = 0, 1            # question types, drawn 50/50
ANSWER, TOOL = 0, 1          # actions: answer directly, or call the search tool

# MEAN_REWARD[question type][action]: the reward each choice earns ON AVERAGE.
MEAN_REWARD = torch.tensor([
    [0.2, 1.0],              # HARD: answering directly is poor, searching is good
    [0.8, 0.5],              # EASY: answering directly is good, searching wastes time
])
NOISE_STD = 1.5              # each rollout's reward is its average plus this much noise


class Policy(nn.Module):
    """pi_theta(action | question type): a 2x2 table of logits. Starts at p(tool) = 0.40."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([[0.6, 0.4], [0.6, 0.4]]).log())

    def dist(self, qtype):
        """The action distribution for each question in `qtype` (a batch of type ids)."""
        return torch.distributions.Categorical(logits=self.logits[qtype])

    @torch.no_grad()
    def probs(self):
        """Current action probabilities, [question type, action]."""
        return torch.softmax(self.logits, dim=-1)

    def true_reward(self):
        """J(theta) of the current policy, as a float. Uses YOUR expected_reward."""
        return float(expected_reward(self.probs()))


@torch.no_grad()
def collect_rollouts(policy, n):
    """Sample n questions, let the policy act, score each action. Uses YOUR compute_advantage."""
    qtype = torch.randint(0, 2, (n,))
    action = policy.dist(qtype).sample()
    reward = MEAN_REWARD[qtype, action] + NOISE_STD * torch.randn(n)
    advantage = compute_advantage(reward)
    old_logp = policy.dist(qtype).log_prob(action)
    return qtype, action, advantage, old_logp


# The running example: 10 HARD questions, 4 tool calls (A = +3), 6 direct answers (A = -2).
EXAMPLE_BATCH = (
    torch.zeros(10, dtype=torch.long),                  # qtype: all HARD
    torch.tensor([TOOL] * 4 + [ANSWER] * 6),            # action
    torch.tensor([3.] * 4 + [-2.] * 6),                 # advantage
    torch.tensor([0.4] * 4 + [0.6] * 6).log(),          # old_logp: what the collecting model gave them
)


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
        got = [round(x, 3) for x in got.detach().flatten().tolist()]
        got = got[0] if len(got) == 1 else got
    print(f"  {label:<44} yours: {str(got):<28} expected: {expected}")


if __name__ == "__main__":
    print("Your functions on the running example (fill a stage, rerun, compare):\n")
    _show("stage 1  expected_reward(start policy)",
          lambda: expected_reward(torch.tensor([[0.6, 0.4], [0.6, 0.4]])), 0.6)
    _show("stage 1  expected_reward(best policy)",
          lambda: expected_reward(torch.tensor([[0.0, 1.0], [1.0, 0.0]])), 0.9)
    _show("stage 2  compute_advantage([1, 2, 3, 6])",
          lambda: compute_advantage(torch.tensor([1., 2., 3., 6.])), [-2.0, -1.0, 0.0, 3.0])
    _show("stage 3  pg_loss(example batch)",
          lambda: pg_loss(Policy(), *EXAMPLE_BATCH), 0.487)

    def one_update():
        policy = Policy()
        vpg_update(policy, torch.optim.SGD(policy.parameters(), lr=0.1), EXAMPLE_BATCH)
        p = policy.probs()[HARD, TOOL]
        return "not done yet (the model did not move)" if abs(p - 0.4) < 1e-6 else p
    _show("stage 4  p(tool|HARD) after one update", one_update, "0.459 (up from 0.40)")
    print("\nWhen these match, run:  python VPG/from_scratch/check.py")
