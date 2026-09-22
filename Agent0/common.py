"""Agent0's self-consistency reward, curation filter, and ADPO.

Agent0 trains two models against each other, with a different algorithm each:

  Curriculum Agent   writes questions    -> GRPO, plain and unmodified (Step 3)
  Executor Agent     solves them, with   -> ADPO, this paper's own variant
                     a Python sandbox       (Step 5). Also called the solver.

Only ADPO is implemented below; the Curriculum Agent's half of the loop uses
``GRPO/common.py`` as it stands. What this module contributes to that half is
the reward -- Agent0's questions have no answer key, so the reward has to come
from the Executor agreeing with itself.

Agent0's pipeline is numbered 1-5, and the step numbers below refer to it:

  Steps 1-2  setup: two conda envs, deploy the SandboxFusion code service
  Step 3     TRAIN the Curriculum Agent. It proposes a question; a frozen
             Executor attempts it 10x with the sandbox; the attempts are voted
             on and gated against the proposer's own claimed answer; GRPO turns
             that reward into a weight update. No dataset is produced.
  Step 4     CURATE. Nothing is trained. The frozen, just-trained Curriculum
             Agent mass-generates ~8,000 questions; a frozen Executor answers
             each 9x by plain generation -- no sandbox, no gate -- and its
             majority answer BECOMES the label. The agreement fraction becomes
             that question's difficulty. Rows inside [0.3, 0.8] are kept.
  Step 5     TRAIN the Executor on what Step 4 kept, with ADPO.

Steps 3 and 4 both vote on repeated Executor attempts, and they are separate
runs of separate code with different sampling -- which is why
:func:`self_consistency_score` needs a ``denominator`` argument at all.

Traced from the real code in `aiming-lab/Agent0`, not from the paper:

  Step 3 reward      curriculum_train/examples/reward_function/curriculum_reward.py
  Step 3 grading     curriculum_train/vllm_service_init/start_vllm_server_tool.py
  Step 4 vote        curriculum_train/question_evaluate/evaluate.py
  Step 4 filter      curriculum_train/question_evaluate/upload.py
  Step 5 reward      executor_train/verl_tool/workers/reward_manager/reward_score/torl_math.py
  Step 5 algorithm   executor_train/verl_tool/trainer/ppo/core_algos.py

vLLM sampling, the SandboxFusion code service, and ``mathruler``'s sympy grader
are represented here by caller-supplied callbacks or by already-sampled values.
Everything through ``difficulty_filter`` is plain Python, exactly as it is
upstream; only ADPO needs torch.
"""

import importlib.util
import re
from collections import Counter
from math import exp, log
from pathlib import Path

import torch


def _load(name, path):
    """Import a module by path. Every directory here has a file named common.py."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ADPO reuses verl's shared loss machinery unchanged; only the clip bound differs.
_ppo = _load("ppo_common", Path(__file__).resolve().parent.parent / "PPO" / "common.py")

# ---------------------------------------------------------------- extraction


def extract_boxed(text):
    """Return the contents of the last ``\\boxed{...}``, or ``""``.

    Braces are walked rather than matched with a regex because the payload
    nests: ``\\boxed{\\frac{1}{2}}`` closes at its fourth ``}``, not its first.
    ``mathruler.extract_boxed_content`` scans the same way, and ``torl_math``'s
    ``boxed_pattern`` only tolerates nesting at all by spelling three levels of
    it out by hand.
    """
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return ""
    depth = 0
    for position in range(start + len(marker) - 1, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(marker):position]
    return ""


def extract_question(text):
    """Return the last ``<question>...</question>`` body, or ``""``.

    ``compute_score`` takes the last match and pairs it with the boxed answer;
    a generation missing either one is the malformed case that
    :func:`curriculum_reward` scores ``-1``.

    .. warning::

       Upstream has a bug at exactly this pairing, and this module does **not**
       reproduce it. ``compute_score`` writes::

           questions = re.findall(r"<question>(.*?)</question>", predicts[i], ...)
           answers   = extract_boxed_content(predicts[i])
           question  = questions[-1].strip()
           answer    = answers[-1].strip()

       ``re.findall`` returns a list, so ``questions[-1]`` is the last match and
       is correct. But ``mathruler.extract_boxed_content`` is typed
       ``(text: str) -> str`` and returns a **string**, so ``answers[-1]`` is
       its **last character**. A proposer claiming ``\boxed{40}`` ships ``"0"``
       as its golden answer.

       Every other call site in the repo (``evaluate.py``, the grading server,
       and ``accuracy_reward`` a few lines above) uses the return value directly
       as a string; this one line treats it like the list beside it. The
       consequence is that :func:`gate_against_claim` compares the Executor's
       majority against a single character, so the gate fails for any
       multi-character answer and the reward collapses to
       ``-cluster_share + tool_reward``.

       This repo pairs the full boxed string, which is what the code clearly
       intends. If you are reproducing upstream numbers, expect the gate to pass
       far more often here than it does there.
    """
    questions = re.findall(r"<question>(.*?)</question>", text, re.DOTALL)
    return questions[-1].strip() if questions else ""


def default_equivalent(first, second):
    """Stand in for ``mathruler.grade_answer``: are two answers the same?

    The real grader normalizes LaTeX and then asks sympy whether the difference
    simplifies to zero, which is what lets ``"1/2"``, ``"0.5"`` and
    ``"\\frac{1}{2}"`` land in one bucket. This keeps the same contract on a
    small set of cases so the exercises stay dependency-free. Every function
    below takes ``equivalent`` as an argument, so the real grader drops in.
    """
    if not first or not second:
        return False
    left, right = _normalize(first), _normalize(second)
    if left == right:
        return True
    try:
        return abs(_to_number(left) - _to_number(right)) < 1e-6
    except ValueError:
        return False


def _normalize(answer):
    answer = answer.strip().lower()
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("$", "").replace(" ", "").replace(",", "")
    return answer.rstrip(".")


def _to_number(answer):
    """Parse a plain number, a ``a/b``, or a ``\\frac{a}{b}``."""
    fraction = re.fullmatch(r"\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", answer)
    if fraction:
        return float(fraction.group(1)) / float(fraction.group(2))
    ratio = re.fullmatch(r"(-?[\d.]+)/(-?[\d.]+)", answer)
    if ratio:
        return float(ratio.group(1)) / float(ratio.group(2))
    return float(answer)


# ------------------------------------------------------------ self-consistency


def cluster_answers(answers, equivalent=default_equivalent):
    """Group answers into equivalence classes, in first-seen order.

    This is ``consolidate_and_grade``'s grouping loop. Three details are load
    bearing:

    * Empty extractions are skipped, never bucketed. They still count against
      Step 3's denominator -- see :func:`self_consistency_score`.
    * A literal ``==`` (and the ``'no '`` in both strings shortcut) is tried
      before ``equivalent``, because the real ``grade_answer`` calls sympy and
      is slow enough upstream to need a 20-second timeout.
    * ``equivalent`` is tried in *both* directions. It is not symmetric: the
      grader normalizes its two arguments differently, so ``f(a, b)`` can be
      false while ``f(b, a)`` is true.

    First match wins; a candidate matching no existing bucket opens its own.
    """
    counts = {}
    for answer in answers:
        if not answer:
            continue
        matched = False
        for existing in list(counts):
            cheap = answer == existing or ("no " in answer.lower() and "no " in existing.lower())
            if cheap or equivalent(answer, existing) or equivalent(existing, answer):
                counts[existing] += 1
                matched = True
                break
        if not matched:
            counts[answer] = 1
    return counts


def self_consistency_score(answers, equivalent=default_equivalent,
                           denominator="candidates"):
    """Return ``(majority_answer, agreement_fraction)`` over sampled answers.

    ``answers`` are already-extracted strings; ``""`` means that candidate
    produced no parseable ``\\boxed{}``.

    The two call sites upstream disagree about the denominator, so it is an
    argument here rather than a constant:

    ``"candidates"``
        Step 3. ``consolidate_and_grade`` divides by ``len(assistant_messages)``
        -- every candidate the Executor was asked for, including the ones that
        answered with nothing. Failing to produce an answer counts as
        disagreement, which is what you want when grading how hard a question is.

    ``"valid"``
        Step 4. ``evaluate.py`` filters ``results`` down to non-empty
        extractions first and divides by what is left. A question the Executor
        usually fails to even answer can still score 1.0 here, on the strength
        of the two attempts that produced anything.

    Same phrase, same repo, two different numbers. The Step 4 value is the one
    that survives into ``train.parquet`` and becomes ADPO's difficulty label.
    """
    if denominator not in ("candidates", "valid"):
        raise ValueError("denominator must be 'candidates' or 'valid'")
    counts = cluster_answers(answers, equivalent)
    if not counts:
        return "", 0.0
    majority = max(counts, key=counts.get)
    total = len(answers) if denominator == "candidates" else sum(
        1 for answer in answers if answer)
    return majority, counts[majority] / total


def gate_against_claim(majority, claimed, score, equivalent=default_equivalent,
                       floor=0.1):
    """Return ``score`` if the Executor's consensus backs the proposer's claim.

    The Curriculum Agent publishes its own ``\\boxed{}`` answer alongside each
    question it writes. Nobody verifies that claim -- but the Executor's
    majority has to land on it, or the whole question scores zero no matter how
    consistent that majority was. Without this gate the proposer could farm
    reward by writing questions the Executor confidently gets wrong.

    ``floor`` is upstream's ``score > 0.1``, and is strictly greater on
    purpose. With the 10 candidates Step 3 samples, ten distinct answers give
    every bucket a count of 1 and an arbitrary "majority" at exactly ``0.1``.
    That is total incoherence, not a 10% consensus, so it is excluded.
    """
    if score > floor and equivalent(majority, claimed):
        return score
    return 0.0


# -------------------------------------------------------------- reward shaping


def tool_reward(predict, weight=0.05, cap=4):
    """Reward Python calls by the count, up to a cap.

    ``calculate_tool_reward`` counts occurrences of the ```` ```output ````
    fence and pays ``weight`` each up to ``cap``. Worth up to 0.20, not the flat
    0.05 bonus the shape is usually described as, and a capped count rather than
    a flag so that using the tool twice beats using it once without paying for
    an unbounded loop of trivial calls.

    **Whose text is counted matters, and it is not the obvious one.**
    ``compute_score`` calls this on ``predicts[i]`` -- the *Curriculum Agent's
    own generation*, the text containing ``<question>`` -- not on the Executor's
    solving transcript. So the term rewards the **proposer** for reaching for
    the sandbox while it composes and checks a question, not the solver for
    using one while answering. Pass the proposer's generation here.
    """
    if not predict:
        return 0.0
    return min(predict.count("```output"), cap) * weight


def curriculum_reward(score, has_question, cluster_share, predict):
    """Assemble Step 3's reward for one generated question.

        (min(score, 1 - score) if has_question else -1.0)
            - cluster_share
            + tool_reward(predict)

    ``min(score, 1 - score)`` is a tent peaking at 0.5: a question the Executor
    solves every time scores 0, and so does one it never solves. Only genuine
    50/50 difficulty pays.

    ``has_question`` false means nothing parsed out of the generation, and the
    base drops to ``-1.0`` -- strictly worse than a perfectly solved question's
    0.0. Malformed output is punished, not merely ignored.

    ``cluster_share`` is subtracted raw, and is a fraction rather than a flag:
    see :func:`bleu_cluster_share`.
    """
    base = min(score, 1.0 - score) if has_question else -1.0
    return base - cluster_share + tool_reward(predict)


def bleu_cluster_share(questions, distance_threshold=0.5):
    """Return each question's share of the batch that looks like it.

    ``cluster_share_per_problem`` builds a ``1 - sentence_bleu`` distance matrix
    over the batch, runs average-linkage agglomerative clustering at
    ``distance_threshold``, and hands back ``cluster_size / batch_size`` per
    row. :func:`curriculum_reward` subtracts that number directly.

    So the diversity penalty is continuous, not the fixed ``0.3 if duplicate``
    it is usually summarized as. One question in a cluster of 40 out of 100
    loses 0.40 -- more than the entire 0.5 maximum of the tent reward. A
    question with no near-duplicates still loses ``1/n``, its own share. The
    pressure is on the batch's variety as a whole, not on exact repeats.

    Upstream uses nltk and scikit-learn; the BLEU and merge loops below are
    written out so this module stays dependency-free.
    """
    if not questions:
        return []
    size = len(questions)
    tokens = [question.split() for question in questions]
    distance = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            value = 1.0 - _sentence_bleu(tokens[j], tokens[i])
            distance[i][j] = distance[j][i] = value

    clusters = _average_linkage(distance, distance_threshold)
    label_of = {}
    for label, members in enumerate(clusters):
        for member in members:
            label_of[member] = label
    sizes = Counter(label_of.values())
    return [sizes[label_of[i]] / size for i in range(size)]


def _sentence_bleu(reference, hypothesis, max_n=4):
    """BLEU-4 with nltk's ``SmoothingFunction().method1`` add-epsilon smoothing."""
    if not hypothesis:
        return 0.0
    log_precisions = []
    for n in range(1, max_n + 1):
        hypothesis_ngrams = Counter(_ngrams(hypothesis, n))
        reference_ngrams = Counter(_ngrams(reference, n))
        total = sum(hypothesis_ngrams.values())
        if total == 0:
            # method1 leaves a precision with no denominator out entirely.
            continue
        overlap = sum(min(count, reference_ngrams[gram])
                      for gram, count in hypothesis_ngrams.items())
        # method1: a zero numerator becomes epsilon rather than collapsing BLEU.
        precision = (overlap / total) if overlap else (1e-9 / total)
        log_precisions.append(log(precision))
    if not log_precisions:
        return 0.0
    score = exp(sum(log_precisions) / max_n)
    if len(hypothesis) < len(reference):
        score *= exp(1.0 - len(reference) / len(hypothesis))
    return score


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _average_linkage(distance, threshold):
    """Merge clusters while the closest pair's mean distance is below threshold."""
    clusters = [[i] for i in range(len(distance))]
    while len(clusters) > 1:
        best, best_pair = None, None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                mean = sum(distance[a][b] for a in clusters[i] for b in clusters[j])
                mean /= len(clusters[i]) * len(clusters[j])
                if best is None or mean < best:
                    best, best_pair = mean, (i, j)
        if best is None or best >= threshold:
            break
        i, j = best_pair
        clusters[i] = clusters[i] + clusters[j]
        clusters.pop(j)
    return clusters


# ------------------------------------------------------------------ curation


def difficulty_filter(rows, min_score=0.3, max_score=0.8):
    """Keep rows whose Step 4 score falls inside an inclusive band.

    ``upload.py``'s whole curation rule: a non-empty answer and
    ``min_score <= score <= max_score``. A flat keep/discard box, not a shaped
    preference -- 0.5 gets no bonus for sitting in the middle. Everything kept
    carries its ``score`` forward as ADPO's difficulty label.

    The defaults here are ``[0.3, 0.8]``, which is what the README's documented
    invocation passes. ``upload.py``'s own argparse default for ``--max_score``
    is ``0.7``; run the script bare and you train on a narrower band than the
    reproduction instructions describe.
    """
    return [row for row in rows
            if row.get("answer") and min_score <= row.get("score", 0) <= max_score]


# -------------------------------------------------- Step 5: Executor and ADPO


def correctness_score(response, ground_truth, equivalent=default_equivalent):
    """Return ``+1.0`` for a correct final answer, ``-1.0`` otherwise.

    ``correctness_score_default``: pull the last ``\\boxed{}``, compare against
    the pseudo-label Step 4 attached, and pay ``+1``/``-1``. A response with no
    boxed answer scores ``-1`` -- the same as being wrong, no separate format
    penalty. That is the entire Executor reward; all of ADPO's difficulty
    awareness lives in the algorithm below, not here.
    """
    answer = extract_boxed(response)
    if not answer:
        return -1.0
    return 1.0 if equivalent(answer, ground_truth) else -1.0


def adpo_trust_scale(score, min_score=0.3, max_score=0.8, min_advantage_scale=0.5):
    """Map a difficulty label to the factor ADPO shrinks its advantage by.

    ``trust = clamp((score - min) / (max - min), 0, 1)`` runs 0 at the hardest
    kept question to 1 at the easiest, and the returned scale interpolates from
    ``min_advantage_scale`` up to 1.0. A hard question teaches a half-strength
    lesson because its reward is the one least likely to be right.
    """
    score = torch.as_tensor(score, dtype=torch.float32)
    trust = ((score - min_score) / (max_score - min_score)).clamp(0.0, 1.0)
    return min_advantage_scale + trust * (1.0 - min_advantage_scale)


def adpo_advantage(token_level_rewards, response_mask, index, score,
                   epsilon=1e-6, norm_adv_by_std_in_grpo=True, min_score=0.3,
                   max_score=0.8, min_advantage_scale=0.5):
    """Return GRPO's group-relative advantage, scaled by per-sample trust.

    Same signature and shapes as ``compute_adpo_outcome_advantage``, which is
    ``compute_grpo_outcome_advantage`` line for line -- same grouping by uid,
    same singleton mean-0/std-1 special case, same ``1e-6``, same sample std --
    plus one multiply at the end. Read it beside ``GRPO/common.py``'s
    ``compute_grpo_outcome_advantage``: the diff is ADPO's whole contribution.

    ``score`` is the extra input, one per row: the Step 4 difficulty label.
    Getting it here at all required hand-editing verl's own ``compute_advantage``
    dispatcher, because the registry has no generic way to know an estimator
    wants a per-sample label pulled out of the batch.

    ``norm_adv_by_std_in_grpo=False`` drops the std divide, exactly as Dr.GRPO's
    one-flag change does.
    """
    scores = token_level_rewards.sum(dim=-1)
    scale = adpo_trust_scale(score, min_score, max_score, min_advantage_scale)

    with torch.no_grad():
        id2score = {}
        for i in range(scores.shape[0]):
            id2score.setdefault(index[i], []).append(scores[i])

        id2mean, id2std = {}, {}
        for idx, group in id2score.items():
            if len(group) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(group) > 1:
                id2mean[idx] = torch.mean(torch.tensor(group))
                id2std[idx] = torch.std(torch.tensor(group))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(scores.shape[0]):
            if norm_adv_by_std_in_grpo:
                adv = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                adv = scores[i] - id2mean[index[i]]
            scores[i] = adv * scale[i]          # <- the entire ADPO twist

        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def adpo_clip_high(score, base_epsilon=0.2, max_epsilon_bonus=0.1,
                   min_score=0.3, max_score=0.8):
    """Return the per-sample upper clip width, widened on harder questions.

    The mirror image of the trust weight: ``exploration`` is 1 at the hardest
    kept question and 0 at the easiest, so the high bound runs from
    ``base_epsilon + max_epsilon_bonus`` down to ``base_epsilon``. The lower
    bound never moves. On a hard question the policy gets more room to raise a
    good-but-unlikely response's probability, because "correct" there does not
    yet mean the approach generalizes.

    DAPO's clip-higher is the same idea with a constant instead of a dial.
    """
    score = torch.as_tensor(score, dtype=torch.float32)
    exploration = ((max_score - score) / (max_score - min_score)).clamp(0.0, 1.0)
    return base_epsilon + exploration * max_epsilon_bonus


def adpo_policy_loss(old_log_prob, log_prob, advantages, response_mask, score,
                     base_epsilon=0.2, max_epsilon_bonus=0.1, min_score=0.3,
                     max_score=0.8, loss_agg_mode="seq-mean-token-mean"):
    """PPO's clipped loss with a PER-SAMPLE upper clip bound.

    Returns ``(pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower)``, matching
    ``compute_policy_loss_adpo``'s contract and PPO's.

    Identical to ``PPO/common.py``'s ``compute_policy_loss`` except that the
    upper bound is a column vector from :func:`adpo_clip_high` rather than one
    shared scalar. The lower bound never moves, and ADPO has no dual clip, so
    ``pg_clipfrac_lower`` is returned as a constant zero -- upstream does the
    same.
    """
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)
    high = 1.0 + adpo_clip_high(score, base_epsilon, max_epsilon_bonus,
                                min_score, max_score).unsqueeze(-1)

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = _ppo.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    # Two steps because the lower bound is a scalar and the upper is per-sample;
    # torch.clamp will not mix the two.
    pg_losses2 = -advantages * torch.minimum(ratio.clamp_min(1.0 - base_epsilon), high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    pg_loss = _ppo.agg_loss(loss_mat=pg_losses, loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode)
    pg_clipfrac = _ppo.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.0)
