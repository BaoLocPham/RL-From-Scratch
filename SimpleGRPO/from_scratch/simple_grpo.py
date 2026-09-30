"""GRPO from scratch, the simple way: the DeepSeekMath algorithm on the multi-step toy.

How to work through it:
  1. Fill the TODOs in STAGE 1 below (each TODO is one line).
  2. Try it:    python SimpleGRPO/from_scratch/simple_grpo.py    prints your results next to the expected ones
  3. Check it:  ./scripts/run_simple_grpo.sh check               stops at the first stage that is not right yet
  4. Move on to the next stage.

Where this sits. SimplePPO gave every step its own advantage with a critic and
GAE -- and on that toy the critic barely paid for itself. GRPO (DeepSeekMath,
§4.1) drops the critic. It answers the SAME question several times (a group)
and judges each answer against the others:

    stage 1  the group advantage      (R - group mean) / group std, the same for every turn
    stage 2  the KL to pi_ref         DeepSeekMath's estimator (eq. 4)
    stage 3  the loop                 your SimplePPO clip + beta * KL, K epochs of minibatches
    stage 4  no code                  your GRPO on the toy, and what each choice does

The clip is not rewritten: stage 3 imports YOUR policy_loss from
SimplePPO/from_scratch/simple_ppo.py. GRPO changes the advantage and adds a KL;
it keeps PPO's clip. So finish SimplePPO first (stages 1 and 2 here do not need it).

The toy is SimplePPO's agent, sampled in groups and graded right (+1) or wrong
(-1): see ../group_env.py. Try not to open ../simple_grpo.py (the reference).
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "SimplePPO" / "from_scratch"))
sys.path.insert(1, str(HERE.parent))
from simple_ppo import policy_loss  # noqa: E402  (YOUR SimplePPO clip, stage 2 there)
from group_env import SEARCH, Policy  # noqa: E402,F401


# ============================================================ STAGE 1: the group advantage
def group_advantage(rewards, group_size, eps=1e-6, scale_by_std=True):
    """GRPO's advantage for (episodes, turns) rewards, stored group by group (DeepSeekMath §4.1.2).

    PPO asked a critic "how good is this state?" and GAE turned its answers into
    one advantage per step. GRPO has no critic. It asks: how did this attempt do
    compared with the OTHER attempts at the same question?

        R_i = episode i's total reward                     outcome supervision: one number per episode
        A_i = (R_i - mean(R_group)) / std(R_group)         judged only against its own group

    The rows of `rewards` come group by group: the first `group_size` episodes
    are attempts at question 0, the next `group_size` at question 1, and so on.
    So `.view(-1, group_size)` puts each group in a row of its own.

    Shapes, for the default batch of 2 questions x 8 attempts (see group_env.rollout):

        rewards      (episodes, turns) = (16, 3)   rewards[i, t]: episode i's reward at turn t
                                                   (-0.1 if it searched; +1 or -1 added on turn 2)
        group_size   8: attempts per question, so episodes = questions * group_size = 2 * 8
        returns      (episodes, turns) = (16, 3)   the advantage of every step

    Then every turn of episode i gets the same A_i. The reward says how the
    episode went, not which turn did it -- there is no per-step credit here, the
    thing GAE gave SimplePPO. `scale_by_std=False` skips the divide: that one
    change is Dr.GRPO.

    Worked example, one HARD question, 4 attempts (group_size 4), each search
    costing 0.1, the answer +1 if right and -1 if wrong:
        search, search, skip    right   R =  1.0 - 0.2 =  0.8
        skip,   skip,   skip    wrong   R = -1.0       = -1.0
        search, skip,   skip    wrong   R = -1.0 - 0.1 = -1.1     (one search: a coin flip, lost)
        skip,   search, search  right   R =  1.0 - 0.2 =  0.8
        mean -0.125, std 1.0689 (the sample std, n - 1 = 3)
        A       = [0.865, -0.819, -0.912, 0.865]          each copied to all 3 turns
        Dr.GRPO = [0.925, -0.875, -0.975, 0.925]          the same without the divide
    """
    with torch.no_grad():                               # a target, computed once per batch: no gradient
        # Outcome supervision: add each episode's rewards (the search costs and the answer) into one number.
        # (episodes, turns) -> (episodes,): sum over dim 1, the turns. (16, 3) -> (16,)
        scores = ...                 # TODO stage 1: one total per episode, shape (episodes,)
        # One row per question, one column per attempt.
        # (episodes,) -> (questions, group_size): (16,) -> (2, 8). groups[q, j] is attempt j at question q.
        groups = ...                 # TODO stage 1: reshape scores to (questions, group_size)
        # Better or worse than THIS question's average -- the group mean is the baseline, not a critic.
        # groups.mean(1, keepdim=True) is (questions, 1) = (2, 1): one mean per row, kept as a column.
        # (2, 8) - (2, 1) broadcasts that column across its row, so every attempt loses its OWN
        # question's mean -> (2, 8). Without keepdim the mean is (2,), and (2, 8) - (2,) lines the 2 up
        # with the LAST dim (8): an error -- or, when questions == group_size, the wrong means, silently.
        centred = ...                # TODO stage 1: subtract each row's mean (keepdim=True keeps it a column)
        if scale_by_std:
            # GRPO divides by the row's spread. torch.std's default is the sample (n - 1) std: keep it.
            # groups.std(1, keepdim=True) is (2, 1) as well: (2, 8) / (2, 1) -> (2, 8).
            centred = ...            # TODO stage 1: divide by each row's std plus eps
        # Back to one number per episode, then the same number on every turn:
        # (2, 8) -> reshape(-1, 1) -> (16, 1): one column, in episode order (row 0's 8 attempts, then
        # row 1's -- the order rollout stored them) -> expand_as(rewards) -> (16, 3), the column copied
        # across the 3 turns.
        return ...                   # TODO stage 1: centred as a column, expanded to the shape of rewards


# ============================================================ STAGE 2: the KL to pi_ref (eq. 4)
def kl_penalty(policy, states, actions, ref_logp):
    """KL(pi_theta || pi_ref), by DeepSeekMath's estimator (eq. 4), averaged over the steps.

    GRPO keeps the policy near a frozen reference pi_ref -- on an LLM, the model
    before RL; here, the starting policy -- by adding beta * KL to the loss.
    TRPO computed the exact KL by summing over both actions. An LLM has ~150k
    actions per step, so DeepSeekMath estimates it from the ONE action taken:

        pi_ref / pi_theta - log(pi_ref / pi_theta) - 1          this is "k3"

    With x = pi_ref / pi_theta, x - log x - 1 is never negative and is 0 only at
    x = 1. Its average over actions drawn from pi_theta is exactly the KL.

    `ref_logp` is log pi_ref(a_t | s_t), recorded at rollout. Take log pi_theta
    from the policy, so the penalty carries gradient.

    Shapes, for one minibatch of M steps (M = 16 in grpo_update; each step is one decision):

        states       (M,)  long, state ids 0..11 (see env.state_id)
        actions      (M,)  long, 0 = SKIP, 1 = SEARCH: the action each step took
        ref_logp     (M,)  float, log pi_ref(a_t | s_t)
        policy.dist(states)                     M Categoricals over the 2 actions, one per state
        policy.dist(states).log_prob(actions)   (M,)  log pi_theta(a_t | s_t), the action each one took
        returns      ()    a scalar: the mean over the M steps

    Worked example, pi_theta(search) = 0.6, pi_ref(search) = 0.4:
        took SEARCH: x = 0.4 / 0.6 = 0.667 -> 0.667 + 0.405 - 1 = 0.0721
        took SKIP:   x = 0.6 / 0.4 = 1.5   -> 1.5   - 0.405 - 1 = 0.0945
        averaged by pi_theta: 0.6 * 0.0721 + 0.4 * 0.0945 = 0.0811 = the exact KL
    """
    # Hint: work in logs. log(pi_ref / pi_theta) = ref_logp - log pi_theta; torch.exp undoes the log.
    # (M,) - (M,) -> (M,): one log-ratio per step. Then k3 elementwise, (M,), and .mean() -> ().
    log_ratio = ...          # TODO stage 2: log(pi_ref / pi_theta) for each step's action
    return ...               # TODO stage 2: the mean of exp(log_ratio) - log_ratio - 1


# ============================================================ STAGE 3: the loop
def grpo_update(policy, optimizer, batch, epochs=10, minibatch_size=16, eps=0.2, beta=0.04):
    """The inner loop: `epochs` passes over ONE batch, in minibatches, on DeepSeekMath's eq. 3.

    SimplePPO's ppo_update, minus the critic: no value loss, no entropy bonus,
    and a KL penalty in their place. Negated because optimizers minimise:

        loss = policy_loss + beta * kl_penalty

    policy_loss is your SimplePPO clip, imported at the top of this file.
    Returns each loss term averaged over every update.

    Shapes, for the default batch of 2 questions x 8 attempts:

        batch["states"], batch["actions"]          (episodes, turns) = (16, 3), long
        batch["old_logp"], batch["ref_logp"]       (16, 3), float: log pi_old and log pi_ref of each action
        batch["advantages"]                        (16, 3), float: stage 1's output
        steps[key]     (episodes * turns,) = (48,): the same, flattened, one sample per decision
        idx            (minibatch_size,) = (16,): positions into those 48 (48 = 3 minibatches of 16)
        steps[key][idx]    (16,): this minibatch's samples
        pg, kl, loss   ()  scalars
    """
    # Every step of an episode has its episode's advantage, so time no longer matters:
    # flatten (episodes, turns) -> (steps,). Given.
    steps = {key: batch[key].reshape(-1) for key in ("states", "actions", "old_logp", "ref_logp", "advantages")}
    n = steps["states"].shape[0]
    totals, updates = {"pg_loss": 0.0, "kl": 0.0}, 0
    for _ in range(epochs):
        order = torch.randperm(n)                       # a new shuffle every epoch
        for start in range(0, n, minibatch_size):
            idx = order[start:start + minibatch_size]
            pg = policy_loss(policy, steps["states"][idx], steps["actions"][idx], steps["old_logp"][idx],
                             steps["advantages"][idx], eps)
            # steps["states"][idx], steps["actions"][idx], steps["ref_logp"][idx]: (16,) each -> kl: ()
            kl = ...                 # TODO stage 3: your stage 2 on this minibatch (states, actions, ref_logp)
            loss = ...               # TODO stage 3: the clip loss plus beta times the KL

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            updates += 1                                # bookkeeping for the log
            for name, value in (("pg_loss", pg), ("kl", kl)):
                totals[name] += float(value.detach())
    return {name: total / updates for name, total in totals.items()}


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
    print(f"  {label:<36} yours: {str(got):<30} expected: {expected}")


def _policy_at(p_search):
    policy = Policy()
    with torch.no_grad():
        policy.logits.copy_(torch.tensor([[1 - p_search, p_search]] * 12).log())
    return policy


if __name__ == "__main__":
    rewards = torch.tensor([[-0.1, -0.1, 1.0], [0.0, 0.0, -1.0], [-0.1, 0.0, -1.0], [0.0, -0.1, 0.9]])
    print("Your functions on the worked examples (fill a stage, rerun, compare):\n")
    _show("stage 1  advantages, turn 0", lambda: group_advantage(rewards, 4)[:, 0], [0.865, -0.819, -0.912, 0.865])
    _show("stage 1  every turn the same", lambda: group_advantage(rewards, 4)[0], [0.865, 0.865, 0.865])
    _show("stage 1  Dr.GRPO, turn 0", lambda: group_advantage(rewards, 4, scale_by_std=False)[:, 0],
          [0.925, -0.875, -0.975, 0.925])
    one = (torch.tensor([0]), torch.tensor([SEARCH]), torch.tensor([0.4]).log())
    _show("stage 2  kl_penalty, 0.6 vs ref 0.4", lambda: kl_penalty(_policy_at(0.6), *one), 0.072)
    _show("stage 2  kl_penalty at pi_ref", lambda: kl_penalty(_policy_at(0.4), *one), 0.0)

    def ten_iterations():
        import group_env
        curve, _, _ = group_env.train(sys.modules[__name__], seed=0, iterations=10)
        return f"J 0.388 -> {curve[-1]:.3f}"
    _show("stage 3  10 iterations of GRPO", ten_iterations, "J 0.388 -> 0.725")
    print("\nWhen these match, run:  ./scripts/run_simple_grpo.sh check")
