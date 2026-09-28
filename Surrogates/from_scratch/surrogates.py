"""The candidates for PPO's slot, from scratch.

How to work through it:
  1. Fill the TODOs in STAGE 1 below (each TODO is one line).
  2. Try it:    python Surrogates/from_scratch/surrogates.py    prints your results next to the expected ones
  3. Check it:  python Surrogates/from_scratch/check.py         stops at the first stage that is not right yet
  4. Move on to the next stage.

The loop is given (../loop.py, the paper's Algorithm 1). What goes in its slot
is yours: every function below has the same shape,

    slot(policy, batch, old_probs) -> loss      (minus the objective: optimizers minimise)

    batch      (qtype, action, advantage, old_logp)    qtype is the state s_t
    old_probs  pi_theta_old(. | s_t) for every rollout, shape [batch, 2]

Same running example as VPG and TRPO: 10 rollouts on HARD questions, collected
at p(tool) = 0.40. 4 called the tool (advantage +3); 6 answered directly
(advantage -2).

Try not to open ../surrogates.py (the reference) -- the hints below are enough.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loop import ANSWER, HARD, TOOL, Policy, mean_kl  # noqa: E402,F401


# ============================================================ STAGE 1: L^CPI, the ratio with no limit (eq. 6)
def ratio_of(policy, batch):
    """r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t), one per rollout.

    You wrote this inside TRPO's surrogate. Here it gets its own name, because
    stages 2 and 4 use it too.
    """
    qtype, action, _, old_logp = batch
    # Hint: `policy.dist(qtype).log_prob(action)` is log pi_theta; exp(log a - log b) = a / b.
    return ...               # TODO stage 1: the ratio, one per rollout -> all 1.0 at theta_old


def cpi_loss(policy, batch, old_probs):
    """-L^CPI (eq. 6): minus the mean of ratio * A. The ratio surrogate, with no limit at all.

    Worked example, the model at p(tool) = 0.48 (ratios 1.2 and 0.867):
        L^CPI = (4 * 1.2 * 3 + 6 * 0.867 * -2) / 10 = 0.40,   loss = -0.40
    """
    advantage = batch[2]
    return ...               # TODO stage 1: minus the mean of ratio * advantage -> -0.40 at p(tool) = 0.48


# ============================================================ STAGE 2: the fixed KL penalty (eq. 5)
def kl_penalty_loss(policy, batch, old_probs, beta):
    """-(L^CPI - beta * KL) (eq. 5): moving is allowed, but every nat of KL costs beta.

    The fence of TRPO turned into a fee: one loss, so plain SGD works. KL is
    `mean_kl(policy, qtype, old_probs)` -- TRPO's, imported above.

    Worked example at p(tool) = 0.48: L^CPI = 0.40 and KL = 0.0129, so with beta = 3
        objective = 0.40 - 3 * 0.0129 = 0.3612,   loss = -0.3612
    """
    advantage = batch[2]
    kl = ...                 # TODO stage 2: the mean KL from theta_old (batch[0] is qtype) -> 0.0129 at p(tool) = 0.48
    return ...               # TODO stage 2: minus (mean of ratio * advantage - beta * kl) -> -0.3612 at beta 3


# ============================================================ STAGE 3: the adaptive beta rule (eq. 8)
def adapt_beta(beta, d, d_targ):
    """The paper's §4 rule, run between iterations. d = the KL the last iteration actually moved.

    Moved too little (d < d_targ / 1.5): halve beta, so moving gets cheaper.
    Moved too far    (d > d_targ * 1.5): double beta, so moving gets dearer.
    Otherwise keep it. (The 1.5 and 2 are the paper's own heuristics.)

    Worked example, d_targ = 0.01, so the band is 0.0067 to 0.015:
        d = 0.003  -> too timid -> beta / 2
        d = 0.013  -> inside    -> beta
        d = 0.030  -> too far   -> beta * 2
    """
    if ...:                  # TODO stage 3: moved too little
        return ...           # TODO stage 3: cheaper
    if ...:                  # TODO stage 3: moved too far
        return ...           # TODO stage 3: dearer
    return beta


class AdaptiveKL:
    """L^KLPEN (eq. 8): stage 2's slot plus an AFTER_ITERATION hook running stage 3. Given.

    One object, because the slot and the hook share beta. Use a fresh one per run.
    """

    def __init__(self, d_targ=0.01, beta=1.0):
        self.d_targ, self.beta = d_targ, beta

    def slot(self, policy, batch, old_probs):
        return kl_penalty_loss(policy, batch, old_probs, self.beta)

    def after_iteration(self, policy, batch, old_probs):
        d = float(mean_kl(policy, batch[0], old_probs).detach())   # how far this iteration moved
        self.beta = adapt_beta(self.beta, d, self.d_targ)


# ============================================================ STAGE 4: L^CLIP (eq. 7)
def clip_loss(policy, batch, old_probs, eps=0.2):
    """-L^CLIP (eq. 7): minus the mean of min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A).

    Once a rollout's ratio has moved eps the way its advantage wants, its score
    goes flat and its gradient is 0. The min keeps the penalty when it moved
    the WRONG way, so that is never clipped away.

    Worked example, one tool call with A = +3 (old p = 0.40), eps = 0.2:
        ratio 1.2: min(3.6, 3.6) = 3.6    still rewarded
        ratio 1.5: min(4.5, 3.6) = 3.6    flat: gradient 0, the model stops here
        ratio 0.7: min(2.1, 2.4) = 2.1    moved the wrong way: still penalised
    """
    advantage = batch[2]
    ratio = ratio_of(policy, batch)
    # Hint: `ratio.clamp(lo, hi)` clips; `torch.min(a, b)` takes the smaller, element by element.
    unclipped = ...          # TODO stage 4: ratio * advantage, L^CPI's term
    clipped = ...            # TODO stage 4: the ratio squashed into [1 - eps, 1 + eps], times advantage
    return ...               # TODO stage 4: minus the mean of the smaller of the two


# Stage 5 needs no code: check.py runs your slots through the loop and prints
# the toy's version of the paper's Table 1.


# ============================================================ playground
# The running example: 10 HARD questions, 4 tool calls (A = +3), 6 direct answers (A = -2).
EXAMPLE_BATCH = (
    torch.zeros(10, dtype=torch.long),                  # qtype: all HARD
    torch.tensor([TOOL] * 4 + [ANSWER] * 6),            # action
    torch.tensor([3.] * 4 + [-2.] * 6),                 # advantage
    torch.tensor([0.4] * 4 + [0.6] * 6).log(),          # old_logp: what the collecting model gave them
)
OLD_PROBS = torch.tensor([[0.6, 0.4]] * 10)             # pi_theta_old(. | HARD) for each rollout


def policy_at(p_tool_hard):
    """A policy whose HARD row is (1 - p, p); the EASY row stays at the start, 0.40."""
    policy = Policy()
    with torch.no_grad():
        policy.logits[HARD] = torch.tensor([1 - p_tool_hard, p_tool_hard]).log()
    return policy


def _show(label, fn, expected):
    """Run one of your functions and print it next to the expected value."""
    try:
        got = fn()
    except Exception as exc:                           # unfinished TODOs land here
        got = f"not done yet ({type(exc).__name__})"
    if got is Ellipsis or (isinstance(got, list) and Ellipsis in got):
        got = "not done yet"
    elif isinstance(got, torch.Tensor):
        got = [round(x, 4) + 0.0 for x in got.detach().flatten().tolist()]   # + 0.0: no -0.0
        got = got[0] if len(got) == 1 else got
    print(f"  {label:<40} yours: {str(got):<18} expected: {expected}")


if __name__ == "__main__":
    print("Your functions on the running example (fill a stage, rerun, compare):\n")
    moved = policy_at(0.48)
    _show("stage 1  cpi_loss at p(tool) = 0.48",
          lambda: cpi_loss(moved, EXAMPLE_BATCH, OLD_PROBS), -0.4)
    _show("stage 2  kl_penalty_loss, beta 3",
          lambda: kl_penalty_loss(moved, EXAMPLE_BATCH, OLD_PROBS, beta=3), -0.3612)
    _show("stage 3  adapt_beta, d = .003/.013/.03",
          lambda: [adapt_beta(1.0, d, 0.01) for d in (0.003, 0.013, 0.03)], [0.5, 1.0, 2.0])
    _show("stage 4  clip_loss at p(tool) = 0.48",
          lambda: clip_loss(moved, EXAMPLE_BATCH, OLD_PROBS), -0.4)
    _show("stage 4  clip_loss at p(tool) = 0.60",
          lambda: clip_loss(policy_at(0.60), EXAMPLE_BATCH, OLD_PROBS), -0.48)
    print("\nWhen these match, run:  python Surrogates/from_scratch/check.py")
