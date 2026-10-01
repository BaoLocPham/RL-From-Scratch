"""Agent0's toy: two agents, each a small table of logits. Given code.

Agent0 (https://arxiv.org/abs/2511.16043) trains two agents against each other:

    Curriculum Agent  pi_theta   writes questions          trained with GRPO
    Executor          pi_phi     answers them              trained with ADPO

Everything here is as small as it can be while keeping that loop intact.

QUESTIONS. 5 difficulty levels x 3 variants = 15 questions, ids 0..14. A
question is nothing but its id; its level is id // 3 + 1:

    ids 0, 1, 2   level 1 (easiest)        ids 12, 13, 14   level 5 (hardest)

A 16th output, MALFORMED, stands for a generation with no usable question in
it. It is what the paper's format check R_format catches.

EXECUTOR pi_phi: ONE number, its skill logit. On a level-L question it picks
one of 10 answers; answer 0 is the right one, 1..9 are wrong. Its logits are

    [skill - L, 0, 0, 0, 0, 0, 0, 0, 0, 0]

so p(right | L) = e^(skill - L) / (e^(skill - L) + 9). Every question shares
the one skill, so training on level 3 also helps level 4. That is what lets
the Curriculum Agent's target move up as the Executor improves.

CURRICULUM AGENT pi_theta: 16 logits, one per output (the 15 questions and
MALFORMED). Writing a question = sampling one id.

TOOL CALLS. The paper rewards questions that make the Executor use its code
tool. Here every attempt at a level-L question makes L tool calls: harder
questions need more of them.
"""

import torch
import torch.nn as nn

LEVELS = 5                    # difficulty levels 1..5
VARIANTS = 3                  # questions per level
QUESTIONS = LEVELS * VARIANTS # 15 question ids, 0..14
MALFORMED = QUESTIONS         # output 15: no usable question
OUTPUTS = QUESTIONS + 1       # what the Curriculum Agent can write
ANSWERS = 10                  # possible answers to any question
RIGHT = 0                     # answer 0 is the right one; 1..9 are wrong
START_SKILL = 4.0             # the Executor's skill before any training


def level(question):
    """A question id's difficulty level, 1..5. Works on ints and on tensors."""
    return question // VARIANTS + 1


class Executor(nn.Module):
    """pi_phi: answers a question. Its only parameter is one skill logit."""

    def __init__(self, skill=START_SKILL):
        super().__init__()
        self.skill = nn.Parameter(torch.tensor(float(skill)))

    def logits(self, levels):
        """(n,) levels -> (n, 10) logits: [skill - L, 0, ..., 0]. Answer 0 is right."""
        levels = torch.as_tensor(levels, dtype=torch.float32)
        wrong = torch.zeros(levels.shape[0], ANSWERS - 1)                 # the 9 wrong answers: logit 0
        right = (self.skill - levels).unsqueeze(1)                        # the right answer: skill - L
        return torch.cat([right, wrong], dim=1)

    def dist(self, levels):
        return torch.distributions.Categorical(logits=self.logits(levels))

    @torch.no_grad()
    def answer(self, question, k):
        """k independent attempts at one question: (k,) answers, each 0..9."""
        return self.dist([level(question)] * k).sample()

    @torch.no_grad()
    def p_right(self, lvl):
        """p(right | level), exactly: e^(skill - L) / (e^(skill - L) + 9)."""
        return float(self.dist([lvl]).probs[0, RIGHT])


class Curriculum(nn.Module):
    """pi_theta: writes a question. 16 logits, one per output. Starts uniform."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(OUTPUTS))

    def dist(self):
        return torch.distributions.Categorical(logits=self.logits)

    @torch.no_grad()
    def write(self, n):
        """n questions, sampled from pi_theta: (n,) ids, 0..15."""
        return self.dist().sample((n,))

    def log_prob(self, questions):
        """log pi_theta(x) for each written question: (n,)."""
        return self.dist().log_prob(questions)

    @torch.no_grad()
    def by_level(self):
        """p(writing a level-L question) for L = 1..5, then p(MALFORMED)."""
        probs = self.dist().probs
        return [float(probs[i * VARIANTS:(i + 1) * VARIANTS].sum()) for i in range(LEVELS)] + [float(probs[MALFORMED])]


def tool_calls(question):
    """N_tool: how many times an attempt at this question calls the code tool. Level L -> L calls."""
    return level(question)
