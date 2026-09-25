"""Vanilla policy gradient, PPO paper Section 2.1 (eq. 1-2). Demo: ``python VPG/run_vpg.py``.

    g_hat    = E_t[ grad log pi_theta(a_t | s_t) * A_t ]                   (eq. 1)
    L^PG     = E_t[      log pi_theta(a_t | s_t) * A_t ]                   (eq. 2)

Eq. 1 is the policy gradient: push up the log-probability of actions that did
better than expected (A_t > 0), push down the ones that did worse. Eq. 2 is the
same thing written as a scalar, so autodiff can produce eq. 1 with .backward().

Its flaw: the gradient is a local measurement. It says which way is uphill only
at the model that collected the batch (theta_old), and nowhere else. So you
either use each batch for ONE update (correct, but slow -- rollouts are the
expensive part), or reuse it and walk past where the reading is valid.

This file also holds the toy the demos run on, and later modules reuse: a tool-calling
agent that, for each question, either answers directly or calls a search tool.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------- the toy
# A contextual bandit: one decision per question, no multi-step episode.
HARD, EASY = 0, 1            # question types, drawn 50/50
ANSWER, TOOL = 0, 1          # actions: answer directly, or call the search tool

# MEAN_REWARD[question type][action]: the reward you get ON AVERAGE.
MEAN_REWARD = torch.tensor([
    [0.2, 1.0],              # HARD: answering directly is poor, searching is good
    [0.8, 0.5],              # EASY: answering directly is good, searching wastes time
])

# Each rollout's reward is the mean above plus noise -- like a real rollout, it
# is partly luck. Against a batch of 16 this noise is large, so a single batch
# can easily give a misleading reading (say, "search hurts on HARD questions").
NOISE_STD = 1.5

# Always choosing the best action: 0.5 * 1.0 (HARD -> TOOL) + 0.5 * 0.8 (EASY -> ANSWER).
BEST_REWARD = 0.9


class Policy(nn.Module):
    """pi_theta(action | question type): the model we train.

    theta is just a 2x2 table of logits, one row per question type. The real
    thing would be an LLM; the table is enough to show every effect here.
    """

    def __init__(self):
        super().__init__()
        # Start both rows at p(answer, tool) = (0.60, 0.40): softmax(log p) = p.
        self.logits = nn.Parameter(torch.tensor([[0.6, 0.4], [0.6, 0.4]]).log())

    def dist(self, qtype):
        """The action distribution for each question in `qtype` (a batch of type ids)."""
        return torch.distributions.Categorical(logits=self.logits[qtype])

    @torch.no_grad()
    def probs(self):
        """Current action probabilities, [question type, action]. For printing and KL only."""
        return torch.softmax(self.logits, dim=-1)

    def true_reward(self):
        """J(theta) of the current policy, as a float. See `expected_reward`."""
        return expected_reward(self.probs()).item()


def expected_reward(probs):
    """J(theta): the EXACT expected reward of a policy with these probabilities.

    probs is [question type, action]. We know MEAN_REWARD, so
        J = 0.5 * sum over types of sum_a pi(a|type) * MEAN_REWARD[type][a].
    A real system can never compute this -- it only sees sampled rollouts. We
    use it as a referee, to score every method without evaluation noise.
    """
    per_type = (probs * MEAN_REWARD).sum(dim=-1)                # expected reward on HARD, on EASY
    return 0.5 * per_type.sum()                                 # the two types are 50/50


def compute_advantage(reward):
    """Advantage A_t: was this rollout better (+) or worse (-) than the batch average?

    The batch mean is the simplest baseline. Without it every reward in this toy
    is mostly positive, so every action taken would be pushed up; subtracting
    the mean makes the update ask "better than usual?" instead of "good at all?".
    """
    return reward - reward.mean()


@torch.no_grad()
def collect_rollouts(policy, n):
    """Sample n questions, let the policy act, and score each action.

    This is the EXPENSIVE part of real RL: an LLM writing answers and calling
    tools. Everything in the batch is produced by the model as it is NOW --
    theta_old, the collecting model -- and is frozen from here on.
    """
    qtype = torch.randint(0, 2, (n,))                           # which questions came in
    action = policy.dist(qtype).sample()                        # what theta_old chose
    reward = MEAN_REWARD[qtype, action] + NOISE_STD * torch.randn(n)
    advantage = compute_advantage(reward)                       # computed ONCE, never updated
    # log pi_theta_old(a_t | s_t): who collected the batch. VPG never looks at it;
    # TRPO and PPO need it to measure how far the model has moved since.
    old_logp = policy.dist(qtype).log_prob(action)
    return qtype, action, advantage, old_logp


# ---------------------------------------------------------------- vanilla PG
def pg_loss(policy, qtype, action, advantage, old_logp):
    """-L^PG (eq. 2), negated because PyTorch optimizers MINIMISE.

    `old_logp` is accepted but unused on purpose: L^PG has no record of the
    model that collected the batch, so it cannot tell a fresh batch from a stale one.
    """
    logp = policy.dist(qtype).log_prob(action)                  # log pi_theta: the CURRENT model
    return -(logp * advantage).mean()                           # its gradient is -g_hat (eq. 1)


def vpg_update(policy, optimizer, batch, updates=1):
    """Take `updates` optimizer steps on one batch.

    updates=1 is vanilla PG: the gradient is read at theta_old, exactly where the
    batch was collected, so it is valid. Every update after the first is taken
    at a model that has already moved, while the batch has not -- the flaw.
    """
    for _ in range(updates):
        loss = pg_loss(policy, *batch)
        optimizer.zero_grad()                                   # clear the previous update's gradient
        loss.backward()                                         # the reading: gradient at the current theta
        optimizer.step()                                        # the update: the model moves, the batch does not


# ---------------------------------------------------------------- demo
def fresh_batch_each_update(seed, updates, lr=0.3, n=16):
    """Vanilla PG used CORRECTLY: collect 16 NEW rollouts before every update.

    Returns (update, probabilities, true J) after every update, starting at 0.
    """
    torch.manual_seed(seed)
    policy = Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=lr)
    rows = [(0, policy.probs(), policy.true_reward())]
    for update in range(1, updates + 1):
        batch = collect_rollouts(policy, n)                     # fresh rollouts, from the current model
        vpg_update(policy, optimizer, batch)                    # one update, then the batch is discarded
        rows.append((update, policy.probs(), policy.true_reward()))
    return rows


def reuse_one_batch(seed, updates, lr=0.3, n=16):
    """The tempting shortcut: collect ONE batch at theta_old, then run `updates` epochs on it.

    Each epoch is one update here (the whole batch, no minibatches). Returns
    (epoch, probabilities, true J) after every epoch, starting at 0.
    """
    torch.manual_seed(seed)
    policy = Policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=lr)     # plain SGD: step = lr * gradient
    batch = collect_rollouts(policy, n)                         # collected ONCE, never refreshed
    rows = [(0, policy.probs(), policy.true_reward())]
    for epoch in range(1, updates + 1):
        vpg_update(policy, optimizer, batch)                    # same 16 rollouts every epoch
        rows.append((epoch, policy.probs(), policy.true_reward()))
    return rows


def print_terms():
    """What every column in the demo logs means."""
    print("""Terms
  p(tool|HARD)  chance the policy calls the search tool on a HARD question. Ideal: 1.0
                (search helps: reward 1.0 vs 0.2 for answering directly).
  p(tool|EASY)  chance it calls the tool on an EASY question. Ideal: 0.0
                (search wastes time: reward 0.5 vs 0.8 for answering directly).
                Both start at 0.40. Good training pushes HARD up and EASY down.
  true J        J(theta) = E[R]: the policy's expected reward over infinitely many
                questions, computed exactly (the mean rewards are known). Training
                never sees it; it only scores the result.
                  start: HARD 0.6*0.2 + 0.4*1.0 = 0.52, EASY 0.6*0.8 + 0.4*0.5 = 0.68
                         J = (0.52 + 0.68) / 2 = 0.60
                  best:  always search on HARD, always answer on EASY
                         J = (1.0 + 0.8) / 2 = 0.90
  rollout       one question + the action taken + its (noisy) reward. The expensive part.
  batch         16 rollouts, all collected by the same model: theta_old.
  update        one optimizer.step(): the only line that changes the model.
  epoch         one full pass over the batch. Here each update uses the whole batch,
                so 1 epoch = 1 update.
  iteration     collect a fresh batch, then run all its epochs on it.
""")
