"""G-Zero's toy: two agents, each a small table of logits. Given code.

G-Zero (https://arxiv.org/abs/2605.09959) trains two agents with no data and
no verifier:

    Proposer   pi_P   picks a query q and a hint h     trained with GRPO
    Generator  pi_G   answers q, with or without h     trained with DPO

QUERIES. 6 query ids, 0..5. Each has a GOOD answer of T = 3 tokens (vocab 4).
The good answers exist only so we can MEASURE progress; training never looks
at them. A hint "knows" the good token at a position, as a teacher would.

GENERATOR pi_G: a table of logits [query, position, token]. Tokens are chosen
independently at each position, so pi_G(a_t | q, a_<t) = pi_G(a_t | q).
It starts out with BLIND SPOTS: positions where it prefers a wrong token.

    query 0: no blind spot            query 3: positions 0 and 2
    query 1: position 1               query 4: positions 1 and 2
    query 2: position 2               query 5: all three

    known position:  good token logit 3.0                -> p(good) = 0.87
    blind spot:      one wrong token logit 1.5, good 0   -> p(good) = 0.13

A HINT names the good token at one position (h = 0, 1, 2) or at all three
(h = 3). Reading it adds a fixed +kappa = 3.0 to that good token's logit:

    logits(q, h) = logits(q) + kappa * onehot(position, good token)

That is the Generator's in-context ability to use a hint. It is fixed; what
DPO trains is the unassisted table, logits(q).

PROPOSER pi_P: 24 logits, one per (query, hint) pair. Writing = sampling one.
A one-position hint is 120 characters long, the all-positions hint 360.
"""

import torch
import torch.nn as nn

QUERIES = 6
T = 3                          # tokens per answer
VOCAB = 4
HINTS = T + 1                  # hint h = 0, 1, 2: one position; h = 3: all of them
ALL = T                        # the all-positions hint
OUTPUTS = QUERIES * HINTS      # what the Proposer can write: 24 (query, hint) pairs
KAPPA = 3.0                    # how much reading a hint raises the good token's logit
HINT_CHARS = [120, 120, 120, 360]                         # |h|, in characters, for h = 0, 1, 2, 3

GOOD = torch.tensor([[(q + t) % VOCAB for t in range(T)] for q in range(QUERIES)])   # (6, 3): hidden
BLIND = [[], [1], [2], [0, 2], [1, 2], [0, 1, 2]]                                    # blind spots per query


def query_of(output):
    """The query of a Proposer output id."""
    return output // HINTS


def hint_of(output):
    """The hint of a Proposer output id: 0, 1, 2 (one position) or 3 (all)."""
    return output % HINTS


def hint_positions(hint):
    """The positions a hint names."""
    return list(range(T)) if hint == ALL else [hint]


def start_logits():
    """The Generator's starting table: (6, 3, 4) logits, with the blind spots above."""
    logits = torch.zeros(QUERIES, T, VOCAB)
    for q in range(QUERIES):
        for t in range(T):
            if t in BLIND[q]:
                logits[q, t, (GOOD[q, t] + 1) % VOCAB] = 1.5      # prefers a wrong token
            else:
                logits[q, t, GOOD[q, t]] = 3.0                     # knows the good one
    return logits


class Generator(nn.Module):
    """pi_G: answers a query, token by token, with or without a hint."""

    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(start_logits())          # the unassisted logits, (6, 3, 4): what DPO trains

    def logits(self, q, hint=None):
        """(3, 4) logits for query q; with a hint, +kappa on the good token at each named position."""
        logits = self.table[q]
        if hint is None:
            return logits
        boost = torch.zeros(T, VOCAB)
        for t in hint_positions(hint):
            boost[t, GOOD[q, t]] = KAPPA                       # reading the hint
        return logits + boost

    def token_logps(self, q, answer, hint=None):
        """log pi_G(a_t | q[, h]) for each token of an answer: (3,) for answer (3,)."""
        return torch.log_softmax(self.logits(q, hint), -1).gather(1, answer[:, None]).squeeze(1)

    @torch.no_grad()
    def answer(self, q, hint=None):
        """One answer, sampled from pi_G(. | q) or pi_G(. | q, h): (3,) token ids."""
        return torch.distributions.Categorical(logits=self.logits(q, hint)).sample()

    @torch.no_grad()
    def p_good(self):
        """p(good token) at every position of every query, unassisted: (6, 3). Measurement only."""
        return torch.softmax(self.table, -1).gather(2, GOOD[:, :, None]).squeeze(2)


class Proposer(nn.Module):
    """pi_P: picks a (query, hint) pair. 24 logits, starting uniform."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(OUTPUTS))

    def dist(self):
        return torch.distributions.Categorical(logits=self.logits)

    @torch.no_grad()
    def write(self, n):
        """n outputs, sampled from pi_P: (n,) ids; query_of / hint_of decode them."""
        return self.dist().sample((n,))

    def log_prob(self, outputs):
        """log pi_P(output) for each: (n,)."""
        return self.dist().log_prob(outputs)

    @torch.no_grad()
    def by_query(self):
        """p(writing a query), summed over its 4 hints: 6 numbers."""
        return self.dist().probs.view(QUERIES, HINTS).sum(1).tolist()

    @torch.no_grad()
    def by_hint(self):
        """p(writing each hint kind), summed over queries: [position 0, 1, 2, all]."""
        return self.dist().probs.view(QUERIES, HINTS).sum(0).tolist()
