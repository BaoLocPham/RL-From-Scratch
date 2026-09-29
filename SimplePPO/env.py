"""The multi-step toy: the tool-calling agent, now over 3 turns. Given code.

The toy track made ONE decision per question. Here the agent makes three:

    turn 0, 1, 2:  SEARCH (costs 0.1, paid at once) or SKIP
    then:          it answers, and the answer's quality arrives as the final reward

How good the answer is depends on the question and how many searches it took:

                  0 searches  1 search  2 searches  3 searches
    HARD             -1.0        0.0       1.0         1.0       search twice
    EASY              1.0        0.9       0.8         0.7       do not search

So the best policy searches exactly twice on HARD questions and never on EASY
ones: J = 0.5 * (1.0 - 0.2) + 0.5 * 1.0 = 0.90. The final reward is noisy (std
1.0), like a real rollout.

The state at each turn is (question type, turn, searches so far): 6 states per
question type, 12 in all. The policy is a 12 x 2 table of logits, the critic a
table of 12 values -- nothing but lookups, like VPG's 2 x 2 table.

Every episode is exactly 3 turns, so a batch is plain (episodes, 3) tensors:
no padding, no mask, no tokens. That is the only new shape; PPO/ later turns
it into LLM tokens.
"""

import itertools

import torch
import torch.nn as nn

HARD, EASY = 0, 1            # question types, drawn 50/50
SKIP, SEARCH = 0, 1          # actions at every turn
TURNS = 3                    # decisions per episode
SEARCH_COST = 0.1            # paid on the turn the search happens
NOISE_STD = 1.0              # the final reward is its average plus this much noise

# ANSWER_QUALITY[question type][searches so far]: the final reward, on average.
ANSWER_QUALITY = torch.tensor([
    [-1.0, 0.0, 1.0, 1.0],   # HARD: needs two searches
    [1.0, 0.9, 0.8, 0.7],    # EASY: every search only makes it worse
])
BEST_J = 0.9                 # search twice on HARD (1.0 - 0.2), never on EASY (1.0)
N_STATES = 12


def state_id(qtype, turn, searches):
    """(question type, turn, searches so far) -> 0..11. On turn t there are t + 1 possible counts.

    Why not just the question type, as on the toy track? Because the right action
    depends on more: on a HARD question, turn 2 with 1 search so far wants SEARCH
    (one more makes two), but turn 2 with 2 searches wants SKIP. A policy indexed
    by qtype alone could not tell those apart. The ids:

                          turn 0   turn 1   turn 2
        HARD (0/1/2 so far)   0      1 / 2    3 / 4 / 5
        EASY (0/1/2 so far)   6      7 / 8    9 / 10 / 11
    """
    return qtype * 6 + turn * (turn + 1) // 2 + searches


class Policy(nn.Module):
    """pi_theta(action | state): a 12 x 2 table of logits. Starts at p(search) = 0.40 everywhere."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([[0.6, 0.4]] * N_STATES).log())

    def dist(self, state):
        """The action distribution for each state in `state` (a tensor of state ids)."""
        return torch.distributions.Categorical(logits=self.logits[state])


class Critic(nn.Module):
    """V(state): a table of 12 values -- the reward the critic expects from each state on. Starts at 0."""

    def __init__(self):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(N_STATES))

    def forward(self, state):
        return self.v[state]


@torch.no_grad()
def rollout(policy, critic, n):
    """Play n episodes with the policy as it is NOW (theta_old). Every tensor is (n, 3), one column per turn.

    rewards   -0.1 on each turn that searched; the answer's quality (plus noise) added on the last turn
    values    V(s_t) as the critic saw it at collection time
    old_logp  log pi_theta_old(a_t | s_t), frozen
    """
    qtype = torch.randint(0, 2, (n,))
    searches = torch.zeros(n, dtype=torch.long)
    states, actions, old_logp, values = [], [], [], []
    rewards = torch.zeros(n, TURNS)
    for turn in range(TURNS):
        state = state_id(qtype, turn, searches)
        action = policy.dist(state).sample()
        states.append(state)
        actions.append(action)
        old_logp.append(policy.dist(state).log_prob(action))
        values.append(critic(state))
        rewards[:, turn] -= SEARCH_COST * action              # the search is paid for now
        searches = searches + action
    rewards[:, -1] += ANSWER_QUALITY[qtype, searches] + NOISE_STD * torch.randn(n)
    return {"states": torch.stack(states, 1), "actions": torch.stack(actions, 1),
            "old_logp": torch.stack(old_logp, 1), "values": torch.stack(values, 1),
            "rewards": rewards, "qtype": qtype, "searches": searches}


@torch.no_grad()
def true_reward(policy):
    """J(theta), exactly: every one of the 2^3 action sequences, weighted by its probability."""
    probs = policy.logits.softmax(-1)
    total = 0.0
    for qtype in (HARD, EASY):
        for sequence in itertools.product((SKIP, SEARCH), repeat=TURNS):
            chance, searches = 1.0, 0
            for turn, action in enumerate(sequence):
                chance *= float(probs[state_id(qtype, turn, searches), action])
                searches += action
            total += 0.5 * chance * float(ANSWER_QUALITY[qtype, searches] - SEARCH_COST * searches)
    return total


def train(impl, seed, iterations=60, episodes=16, lr=0.3, gamma=1.0, lam=0.8, advantage=None, **update):
    """Algorithm 1, outer loop: collect -> advantages -> ppo_update, `iterations` times.

    `impl` is a module with compute_gae and ppo_update: the reference
    (SimplePPO/simple_ppo.py) or yours. `advantage` replaces GAE for the
    comparisons in run_simple_ppo.py. Returns the true J after every iteration,
    and the trained policy and critic.
    """
    torch.manual_seed(seed)
    policy, critic = Policy(), Critic()
    optimizer = torch.optim.SGD(list(policy.parameters()) + list(critic.parameters()), lr=lr)
    curve = []
    for _ in range(iterations):
        batch = rollout(policy, critic, episodes)                              # theta_old
        if advantage is None:
            batch["advantages"], batch["returns"] = impl.compute_gae(batch["rewards"], batch["values"], gamma, lam)
        else:
            batch["advantages"], batch["returns"] = advantage(batch)
        impl.ppo_update(policy, critic, optimizer, batch, **update)            # K epochs of minibatches
        curve.append(true_reward(policy))
    return curve, policy, critic


def describe(policy):
    """p(search) in every state, as printable rows."""
    probs = policy.logits.softmax(-1)[:, SEARCH].detach()
    rows = []
    for qtype, name in ((HARD, "HARD"), (EASY, "EASY")):
        for turn in range(TURNS):
            cells = [f"{searches} so far: {float(probs[state_id(qtype, turn, searches)]):.2f}"
                     for searches in range(turn + 1)]
            rows.append(f"  {name} turn {turn}:  " + "   ".join(cells))
    return rows
