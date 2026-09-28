"""TRPO from scratch.

How to work through it:
  1. Fill the TODOs in STAGE 1 below (each TODO is one line).
  2. Try it:    python TRPO/from_scratch/trpo.py       prints your results next to the expected ones
  3. Check it:  python TRPO/from_scratch/check.py      stops at the first stage that is not right yet
  4. Move on to the next stage.

Same running example as the VPG exercise: 10 rollouts on HARD questions,
collected at p(tool) = 0.40. 4 called the tool and went well (advantage +3);
6 answered directly and went badly (advantage -2).

Try not to open ../trpo.py (the reference) -- the hints below are enough.

    maximize    mean over the batch of  pi_theta(a|s) / pi_theta_old(a|s) * A     (eq. 3)
    subject to  mean over the batch of  KL[pi_theta_old(.|s), pi_theta(.|s)] <= delta   (eq. 4)

The toy (Policy, collect_rollouts, ...) is VPG's, imported from VPG/vpg.py.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "VPG"))
from vpg import ANSWER, EASY, HARD, TOOL, Policy, collect_rollouts  # noqa: E402,F401


# ============================================================ STAGE 1: the ratio surrogate (eq. 3)
def surrogate(policy, qtype, action, advantage, old_logp):
    """Eq. 3: the mean of ratio * A, with ratio = pi_theta(a|s) / pi_theta_old(a|s).

    Step 1: log pi_theta(a|s) -- exactly as in VPG's pg_loss.
    Step 2: the ratio, new chance / old chance. You only have LOG-probs, and
            e^(log a - log b) = a / b, so subtract first and exponentiate after.
    Step 3: multiply by the advantage and average. No minus sign here: this is
            the objective to go UP. trpo_update (stage 3) negates it.

    Worked example, the running example after the model moved to p(tool) = 0.48:
        tool calls:     0.48 / 0.40 = 1.200,  times A = +3  ->  +3.600   (x4)
        direct answers: 0.52 / 0.60 = 0.867,  times A = -2  ->  -1.733   (x6)
        surrogate = (4 * 3.600 + 6 * -1.733) / 10 = 0.40
    At theta_old itself every ratio is 1, so the surrogate is mean(A) = 0.

    Unlike L^PG, this one USES old_logp: it remembers which model collected the batch.
    """
    # Hint: `policy.dist(qtype).log_prob(action)` gives log pi_theta of the action taken;
    #       `torch.exp(x)` undoes a log.
    logp = ...               # TODO stage 1, step 1: log pi_theta(a|s) per rollout -> shape [10]
    ratio = ...              # TODO stage 1, step 2: pi_theta / pi_theta_old -> 1.0 everywhere at theta_old
    return ...               # TODO stage 1, step 3: the mean of ratio * advantage -> 0.40 at p(tool) = 0.48


# ============================================================ STAGE 2: the KL (eq. 4's left side)
def mean_kl(policy, qtype, old_probs):
    """Mean over the batch of KL(pi_theta_old || pi_theta), computed exactly.

        KL(P || Q) = sum over actions x of  P(x) * (ln P(x) - ln Q(x)),   P = old, Q = new

    ``old_probs`` is P: shape [batch, 2], one row per rollout, [p(answer), p(tool)]
    under theta_old. Q is the CURRENT policy on the same questions.

    KL compares the WHOLE distribution (both actions), not just the action that
    was taken -- which is why it takes old_probs, not old_logp.

    Worked example, old (answer 0.60, tool 0.40) -> new (0.52, 0.48), 10 HARD rollouts:
        answer:  0.60 * (ln 0.60 - ln 0.52) = +0.0859
        tool:    0.40 * (ln 0.40 - ln 0.48) = -0.0729
        KL per rollout = 0.0859 - 0.0729 = 0.0129;  mean over the 10 rollouts = 0.0129
    """
    # Hint: `policy.logits[qtype]` picks each rollout's row of logits, shape [batch, 2];
    #       `torch.log_softmax(x, dim=-1)` turns logits into log-probabilities.
    #       Then `.sum(dim=-1)` adds over the actions and `.mean()` over the batch.
    new_logp = ...           # TODO stage 2, step 1: ln Q(x) for both actions of every rollout -> shape [batch, 2]
    return ...               # TODO stage 2, step 2: sum P * (ln P - ln Q) over actions, mean over the batch -> 0.0129


# ============================================================ STAGE 3: the update, behind the KL fence
def trpo_update(policy, batch, max_kl=0.01, lr=0.3, epochs=50):
    """Maximize eq. 3 subject to eq. 4, on one batch. Returns how many epochs were kept.

    Take up to `epochs` gradient steps on -surrogate, all on this same batch.
    After each step, measure the KL from theta_old. If it is above max_kl, the
    step left the trust region: put the logits back and stop.

    The loop and the optimizer calls are written for you (they are VPG's).
    What is left is the fence: remember where you started, remember where each
    step started, and check the KL after every step.
    """
    qtype = batch[0]
    # Hint: `policy.probs()` is the [question type, action] table, computed without gradient.
    old_probs = ...          # TODO stage 3: pi_theta_old(.|s) for every rollout, frozen -> shape [batch, 2]
    optimizer = torch.optim.SGD(policy.parameters(), lr=lr)
    kept = 0
    for _ in range(epochs):
        # Hint: `.detach().clone()` makes a copy that later steps cannot change.
        before = ...         # TODO stage 3: a copy of policy.logits, to undo this step if needed

        loss = ...           # TODO stage 3: minus your surrogate on this batch  (-surrogate(policy, *batch))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()     # one small step, on the same batch

        if ...:              # TODO stage 3: the KL constraint is broken (your mean_kl, compared with max_kl)
            with torch.no_grad():
                ...          # TODO stage 3: put the logits back to `before`  (policy.logits.copy_(...))
            break            # this batch is used up
        kept += 1
    return kept


# Stage 4 needs no code: check.py runs your trpo_update on the unlucky batch that
# broke VPG (seed 17) and shows what the KL fence saves.


# ============================================================ playground
# The running example: 10 HARD questions, 4 tool calls (A = +3), 6 direct answers (A = -2).
EXAMPLE_BATCH = (
    torch.zeros(10, dtype=torch.long),                  # qtype: all HARD
    torch.tensor([TOOL] * 4 + [ANSWER] * 6),            # action
    torch.tensor([3.] * 4 + [-2.] * 6),                 # advantage
    torch.tensor([0.4] * 4 + [0.6] * 6).log(),          # old_logp: what the collecting model gave them
)


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
    if got is Ellipsis:
        got = "not done yet"
    elif isinstance(got, torch.Tensor):
        got = [round(x, 4) + 0.0 for x in got.detach().flatten().tolist()]   # + 0.0: no -0.0
        got = got[0] if len(got) == 1 else got
    print(f"  {label:<46} yours: {str(got):<28} expected: {expected}")


if __name__ == "__main__":
    print("Your functions on the running example (fill a stage, rerun, compare):\n")
    _show("stage 1  surrogate at theta_old (p = 0.40)",
          lambda: surrogate(Policy(), *EXAMPLE_BATCH), 0.0)
    _show("stage 1  surrogate after moving to p = 0.48",
          lambda: surrogate(policy_at(0.48), *EXAMPLE_BATCH), 0.4)
    old = Policy().probs()[EXAMPLE_BATCH[0]]
    _show("stage 2  mean_kl, 0.40 -> 0.45",
          lambda: mean_kl(policy_at(0.45), EXAMPLE_BATCH[0], old), 0.0051)
    _show("stage 2  mean_kl, 0.40 -> 0.48",
          lambda: mean_kl(policy_at(0.48), EXAMPLE_BATCH[0], old), 0.0129)

    def one_update():
        policy = Policy()
        kept = trpo_update(policy, EXAMPLE_BATCH, max_kl=0.01, lr=0.02)
        return f"{kept} epochs, p(tool|HARD) = {policy.probs()[HARD, TOOL]:.3f}"
    _show("stage 3  trpo_update, lr 0.02, max_kl 0.01", one_update, "5 epochs, p(tool|HARD) = 0.460")
    print("\nWhen these match, run:  python TRPO/from_scratch/check.py")
