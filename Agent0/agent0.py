"""Agent0, the paper's equations on the toy. Demo: ``python Agent0/run_agent0.py``.

Agent0 (https://arxiv.org/abs/2511.16043) has no training data. Two agents
make it for each other, in a loop the paper runs T = 3 times:

    Step 3  train the Curriculum Agent pi_theta (GRPO); the Executor pi_phi is frozen
            pi_theta writes questions; pi_phi answers each one k = 10 times;
            the reward R_C pays for questions pi_phi is unsure about (Eq. 2-5)
    Step 4  curate a dataset; both frozen
            keep the questions pi_phi answers consistently 30-80% of the time (Eq. 7);
            the majority answer y~ becomes the label: nobody knows the true answer
    Step 5  train the Executor pi_phi (ADPO) on that dataset; pi_theta is frozen
            reward 1 for agreeing with y~, 0 otherwise; GRPO's advantage, scaled by
            how much the label can be trusted, s(p^); a clip that is wider on hard questions

Notation follows the paper: x is a question, o_1..o_k or y_1..y_k are the
Executor's k answers to it, p^ (p_hat) is how often they agree, y~ (y_tilde)
is their majority. The toy -- the questions and both agents -- is in agent0_env.py.
"""

from collections import Counter

import torch

from agent0_env import MALFORMED, Curriculum, Executor, level, tool_calls

# ================================================================ the reward pieces


def self_consistency(answers):
    """The Executor's majority answer and how often it was given (Eq. 6).

        p^(x) = (1/k) * sum_i 1(o_i = y~),     y~ = argmax_y sum_i 1(o_i = y)

    answers: (k,) the Executor's k answers to one question x.
    Returns (y_tilde, p_hat): the majority answer, and the fraction of the k that gave it.
    """
    votes = Counter(answers.tolist())               # sum_i 1(o_i = y), for every answer y given
    y_tilde, count = votes.most_common(1)[0]        # y~ = argmax_y; ties go to the answer seen first
    p_hat = count / len(answers)                    # p^ = (1/k) * sum_i 1(o_i = y~): divide by ALL k
    return y_tilde, p_hat


def uncertainty_reward(p_hat):
    """How unsure the Executor is about question x (Eq. 2).

        R_unc(x) = 1 - 2 * |p^(x) - 0.5|

    1 when p^ = 0.5 (it agrees with itself half the time: the question is at its
    frontier), 0 when p^ = 0 or 1 (always the same answer: too easy).
    """
    return 1.0 - 2.0 * abs(p_hat - 0.5)             # R_unc: a tent, peak 1 at p^ = 0.5


def tool_reward(n_tool, gamma=0.05, cap=4):
    """Reward for questions that make the Executor use its tool (Eq. 3).

        R_tool(x) = gamma * min(N_tool, C),     C = 4

    The cap C stops a question earning more just by forcing endless tool calls.
    The paper sets C = 4 but leaves gamma open; gamma = 0.05 is our choice.
    """
    return gamma * min(n_tool, cap)                 # R_tool: gamma per tool call, at most C calls


def repetition_penalty(questions, lambda_rep=1.0):
    """Penalty for writing what the rest of the batch also wrote (Eq. 4).

        R_rep(x_i) = lambda_rep * |C_k| / B,     x_i in cluster C_k

    The paper clusters the batch's questions by BLEU similarity. A question is
    only an id here, so a cluster is simply every copy of the same question.
    The paper leaves lambda_rep open; 1.0 is our choice.

    questions: (B,) the batch of question ids. Returns (B,): each one's penalty.
    """
    cluster_size = torch.bincount(questions)[questions]          # |C_k|: copies of x_i in the batch, (B,)
    return lambda_rep * cluster_size.float() / len(questions)    # R_rep = lambda_rep * |C_k| / B


def curriculum_reward(well_formed, r_unc, r_tool, r_rep, lambda_unc=1.0, lambda_tool=0.6):
    """The Curriculum Agent's reward for each question (Eq. 5).

        R_C(x_i) = R_format(x_i) * max(0, lambda_unc * R_unc(x_i) + lambda_tool * R_tool(x_i) - R_rep(x_i))

    All inputs are (B,). R_format is 1 for a usable question and 0 for a
    malformed one, so a malformed question earns 0 whatever the other terms say.
    max(0, .) keeps every reward non-negative. lambda_tool = 0.6 is the paper's;
    it leaves lambda_unc open, and 1.0 is our choice.
    """
    inner = lambda_unc * r_unc + lambda_tool * r_tool - r_rep    # lambda_unc R_unc + lambda_tool R_tool - R_rep
    return well_formed * torch.clamp(inner, min=0.0)             # R_format * max(0, inner)


def keep(p_hat, low=0.3, high=0.8):
    """Does question x go into the Executor's dataset (Eq. 7)?

        D = { x : p^(x) in [0.3, 0.8] }

    The paper writes this as |p^ - 0.5| <= delta, with delta = 0.25, and states
    the kept range as 0.3 to 0.8; we use the stated range, edges included.
    Below 0.3 the majority is too unreliable to use as a label; above 0.8 the
    question is too easy to teach anything.
    """
    return low <= p_hat <= high                     # inside the band, edges included


def executor_reward(answers, y_tilde):
    """The Executor's reward for each attempt at a curated question.

        R_i = 1(o_i = y~)

    1 for agreeing with the majority label, 0 otherwise. Not the TRUE answer:
    nobody in Agent0 knows it. answers: (G,). Returns (G,) of 0.0 / 1.0.
    """
    return (answers == y_tilde).float()             # R_i = 1(o_i = y~)


# ================================================================ the update


def group_advantage(rewards, eps=1e-6):
    """GRPO's advantage: each answer against the other answers in its group.

        A^_i = (R_i - mean(R_group)) / std(R_group)

    rewards: (groups, G), one row per group (the G answers to one question, or
    the G questions from one prompt). Returns (groups, G).
    """
    mean = rewards.mean(dim=1, keepdim=True)        # mean(R_group), one per row: (groups, 1)
    std = rewards.std(dim=1, keepdim=True)          # std(R_group), sample std: (groups, 1)
    return (rewards - mean) / (std + eps)           # A^_i; a group that all agrees gets 0


def adpo_scale(p_hat):
    """ADPO's trust in a question's label: s(x) = f(p^(x)), f increasing.

        A~_i = A^_i * s(x)

    A low p^ means the majority label y~ is often wrong, so its lesson is
    shrunk. The paper only says f is increasing; we use a straight line over
    the kept band, from 0.5 at p^ = 0.3 up to 1.0 at p^ = 0.8.
    """
    t = min(max((p_hat - 0.3) / 0.5, 0.0), 1.0)     # where p^ sits in [0.3, 0.8], as 0..1
    return 0.5 + 0.5 * t                            # s(x): 0.5 for the hardest kept question, 1.0 for the easiest


def adpo_eps_high(p_hat):
    """ADPO's upper clip bound: eps_high(x), decreasing in p^.

    A hard question (low p^) gets more room to raise a good answer's
    probability. The paper only says eps_high decreases with p^; we use a
    straight line from 0.3 at p^ = 0.3 down to the usual 0.2 at p^ = 0.8.
    """
    t = min(max((0.8 - p_hat) / 0.5, 0.0), 1.0)     # how far below the band's top, as 0..1
    return 0.2 + 0.1 * t                            # eps_high(x): 0.3 for the hardest kept question


def clipped_loss(logp, old_logp, advantages, eps_high, eps_low=0.2):
    """PPO's clipped loss, with an upper bound per sample (Eq. 8, negated to minimise).

        L = -(1/G) * sum_i min( r_i * A~_i,  clip(r_i, 1 - eps_low, 1 + eps_high(x)) * A~_i ),
        r_i = pi(o_i) / pi_old(o_i)

    Trains both agents. The Curriculum Agent (GRPO): plain advantages and
    eps_high = 0.2 for every sample. The Executor (ADPO): advantages times
    s(x), and eps_high(x) per sample. All inputs are (n,).
    """
    ratio = torch.exp(logp - old_logp)                                    # r_i = pi / pi_old
    clipped = torch.minimum(torch.clamp(ratio, min=1.0 - eps_low), 1.0 + eps_high)   # clip(r_i, 1 - eps_low, 1 + eps_high)
    return -torch.min(ratio * advantages, clipped * advantages).mean()    # -(1/G) sum_i min(...)


# ================================================================ the loop


def score_questions(questions, executor, k=10, lambda_rep=1.0):
    """Step 3's reward for a batch of written questions: every term of Eq. 5, each (B,).

    For each question, the frozen Executor answers it k = 10 times, and
    self_consistency turns the answers into p^.
    """
    p_hat = torch.zeros(len(questions))
    for i, question in enumerate(questions.tolist()):
        if question != MALFORMED:                                         # a malformed question has no answers
            _, p_hat[i] = self_consistency(executor.answer(question, k))  # p^(x_i), from k answers
    well_formed = (questions != MALFORMED).float()                        # R_format: 1, or 0 if malformed
    r_unc = torch.tensor([uncertainty_reward(float(p)) for p in p_hat])   # R_unc(x_i)       (Eq. 2)
    r_tool = torch.tensor([tool_reward(tool_calls(q)) for q in questions.tolist()])   # R_tool(x_i)  (Eq. 3)
    r_rep = repetition_penalty(questions, lambda_rep)                     # R_rep(x_i)       (Eq. 4)
    r_c = curriculum_reward(well_formed, r_unc, r_tool, r_rep)            # R_C(x_i)         (Eq. 5)
    return {"p_hat": p_hat, "r_unc": r_unc, "r_tool": r_tool, "r_rep": r_rep, "r_c": r_c}


def train_curriculum(curriculum, executor, optimizer, steps=30, groups=8, group_size=4, k=10, epochs=2,
                     lambda_rep=1.0):
    """Step 3: GRPO on the Curriculum Agent's 16 logits. The Executor is frozen.

    Each step writes B = groups * group_size = 32 questions. A group is the 4
    questions written for one prompt, and GRPO compares them with each other.
    """
    for _ in range(steps):
        with torch.no_grad():
            questions = curriculum.write(groups * group_size)             # x_i ~ pi_theta, (B,)
            r_c = score_questions(questions, executor, k, lambda_rep)["r_c"]   # R_C(x_i), (B,)
            advantages = group_advantage(r_c.view(groups, group_size)).view(-1)   # A^_i within each group of 4
            old_logp = curriculum.log_prob(questions)                     # log pi_theta_old(x_i), frozen
        for _ in range(epochs):
            loss = clipped_loss(curriculum.log_prob(questions), old_logp, advantages,
                                torch.full_like(advantages, 0.2))         # plain GRPO: eps_high = 0.2 for all
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def curate(curriculum, executor, n=64, k=10):
    """Step 4: the frozen Curriculum Agent writes n questions; keep those inside the band (Eq. 7).

    Returns the dataset: one row per kept question, with its majority label y~
    and its p^. Both agents are frozen; nothing learns here.
    """
    dataset = []
    for question in curriculum.write(n).tolist():
        if question == MALFORMED:                                         # nothing to label
            continue
        y_tilde, p_hat = self_consistency(executor.answer(question, k))   # y~ and p^ from k answers (Eq. 6)
        if keep(p_hat):                                                   # Eq. 7
            dataset.append({"question": question, "y_tilde": y_tilde, "p_hat": p_hat})
    return dataset


def train_executor(executor, optimizer, dataset, steps=20, batch=8, group_size=8, epochs=4, adpo=True):
    """Step 5: ADPO on the Executor's skill logit, against the curated labels. The Curriculum Agent is frozen.

    Each step takes `batch` questions from the dataset, samples G = 8 answers
    to each, and scores them against the label y~. adpo=False is plain GRPO:
    no scale s(x), and eps_high = 0.2 for every question.
    """
    if not dataset:
        return
    for _ in range(steps):
        rows = [dataset[i] for i in torch.randint(len(dataset), (batch,)).tolist()]
        levels = torch.tensor([level(row["question"]) for row in rows]).repeat_interleave(group_size)  # (batch * G,)
        y_tilde = torch.tensor([row["y_tilde"] for row in rows]).repeat_interleave(group_size)         # (batch * G,)
        p_hat = [row["p_hat"] for row in rows for _ in range(group_size)]                             # (batch * G,)
        with torch.no_grad():
            answers = executor.dist(levels).sample()                      # o_i ~ pi_phi_old, (batch * G,)
            rewards = executor_reward(answers, y_tilde)                   # R_i = 1(o_i = y~)
            advantages = group_advantage(rewards.view(batch, group_size)).view(-1)   # A^_i, per question
            if adpo:
                advantages = advantages * torch.tensor([adpo_scale(p) for p in p_hat])   # A~_i = A^_i * s(x)
                eps_high = torch.tensor([adpo_eps_high(p) for p in p_hat])               # eps_high(x)
            else:
                eps_high = torch.full_like(advantages, 0.2)
            old_logp = executor.dist(levels).log_prob(answers)            # log pi_phi_old(o_i), frozen
        for _ in range(epochs):
            loss = clipped_loss(executor.dist(levels).log_prob(answers), old_logp, advantages, eps_high)   # Eq. 8
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def agent0(seed=0, iterations=3, adpo=True, lambda_rep=1.0, executor_lr=0.05, report=None):
    """The whole loop: Steps 3, 4 and 5, `iterations` times (the paper's T = 3).

    Returns the trained (curriculum, executor), and one record per iteration.
    `report(iteration, record)`, if given, is called after each iteration.
    """
    torch.manual_seed(seed)
    curriculum, executor = Curriculum(), Executor()
    curriculum_optimizer = torch.optim.SGD(curriculum.parameters(), lr=1.0)
    executor_optimizer = torch.optim.SGD(executor.parameters(), lr=executor_lr)
    history = []
    for iteration in range(1, iterations + 1):
        train_curriculum(curriculum, executor, curriculum_optimizer, lambda_rep=lambda_rep)   # Step 3
        writes = curriculum.by_level()                                    # pi_theta, after Step 3
        dataset = curate(curriculum, executor)                            # Step 4
        skill_before = float(executor.skill.detach())
        train_executor(executor, executor_optimizer, dataset, adpo=adpo)  # Step 5
        record = {"writes": writes, "kept": len(dataset),
                  "labels_right": sum(row["y_tilde"] == 0 for row in dataset),
                  "skill_before": skill_before, "skill": float(executor.skill.detach()),
                  "p_right": [executor.p_right(lvl) for lvl in range(1, 6)],
                  "top_question": float(curriculum.logits.detach().softmax(-1).max())}
        history.append(record)
        if report:
            report(iteration, record)
    return curriculum, executor, history
