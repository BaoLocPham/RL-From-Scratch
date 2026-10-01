# How the three-turn toy works in SimplePPO, SimpleGRPO and SimpleDPO

SimplePPO, SimpleGRPO and SimpleDPO all train an agent on the same small game.
This page explains that game from the ground up:
- what an attempt is;
- exactly how it is rewarded;
- what the agent sees when it decides;
- how each of the three algorithms learns from its attempts.

Every number on this page comes from running the repo's own code.

**The short answer to "how is this different from GRPO?": it isn't.** The game,
the question types and the rewards are the same in all three modules. Two
things change:
- **how many times each question is tried:** once in PPO, 8 times in GRPO,
  twice in DPO;
- **what each attempt is compared with:** a learned guess in PPO, the other
  attempts in GRPO, a single rival attempt in DPO.

---

## Glossary

Skim it now and come back when a word is unclear.

**The game**

| Term | Meaning | In the code |
|---|---|---|
| question type | HARD or EASY, picked 50/50. A question has no text: its type is all there is. | `qtype` (HARD = 0, EASY = 1) |
| turn | one of the agent's three decisions, numbered 0, 1, 2 | `turn` |
| action | what the agent does at a turn: SEARCH (costs 0.1) or SKIP | `actions` (SKIP = 0, SEARCH = 1) |
| attempt | one try at a question: three actions, then the answer. Also called an *episode*. | one row of a batch |
| k | how many of the three turns searched: 0 to 3 | `searches` |
| right / wrong | whether the answer turned out correct | `correct` |

**Rewards**

| Term | Meaning | In the code |
|---|---|---|
| r₀, r₁, r₂ | the reward paid at turn 0, 1 and 2 | `rewards[:, t]` |
| total | r₀ + r₁ + r₂: what one attempt scores | `rewards.sum(1)` |
| p(right) | the chance the answer is right; set by the question type and k | `P_CORRECT` |
| average total | an attempt's total, averaged over right and wrong. Depends only on the question type and k. | `QUALITY` (SimpleDPO) |
| J | the policy's average total over all questions, computed exactly. Every results table reports it; training never sees it. | `true_reward(policy)` |

**The agent**

| Term | Meaning | In the code |
|---|---|---|
| state | what the agent knows before a decision: (question type, turn, searches so far) | `state_id(...)` |
| state id, row | the state's number, 0 to 11: which row of the policy table to use | `states` |
| logits | two raw scores per row; softmax turns them into p(SKIP) and p(SEARCH) | `policy.logits` |
| policy, π | the 12 × 2 table of logits: how the agent decides in every state | `Policy` |
| starting policy | p(SEARCH) = 0.4 in every row: where every run begins | `Policy()` |
| π_ref | a frozen copy of the starting policy. GRPO and DPO measure the policy against it. | `ref_policy` |

**Learning**

| Term | Meaning | In the code |
|---|---|---|
| batch | the 16 attempts collected before each update | `rollout`, `collect_pairs` |
| iteration | collect a batch, learn from it, repeat. Every run does 60. | `train` |
| seed | the starting point of the random numbers. Each result averages 20 seeds. | `torch.manual_seed` |
| baseline | what a typical attempt at this kind of question scores: the yardstick an attempt is judged against | — |
| advantage | how much better than the baseline something did. Positive: make it more likely. Negative: less likely. | `advantages` |
| critic, V(s) | PPO only: a learned guess of the reward still to come from each state | `Critic` |
| GAE | PPO only: its recipe for turning rewards and the critic's guesses into one advantage per turn | `compute_gae` |
| group | GRPO only: all the attempts at the same question (8) | `group_size` |
| pair: chosen, rejected | DPO only: two attempts at one question, the better one and the other | `collect_pairs` |
| label | DPO only: which attempt of a pair is better. Made before training; DPO only ever sees labels. | `labels="better"` |
| β | how strongly GRPO and DPO hold the policy near π_ref | `beta` |

---

## Contents

1. The game in one minute
2. The reward: what each attempt is paid
3. How the agent decides: the state and the policy table
4. The problem all three algorithms solve
5. SimplePPO: one attempt per question, a critic as the yardstick
6. SimpleGRPO: eight attempts per question, the group as the yardstick
7. SimpleDPO: two attempts per question, labelled by which is better
8. How the experiments run
9. From the toy to an LLM

---

## 1. The game in one minute

An agent answers questions. Before answering, it gets **three turns**, and at
each one it chooses:

- **SEARCH**: look something up. It costs 0.1, and on a HARD question it
  makes a right answer more likely.
- **SKIP**: do nothing this turn.

After turn 2 it answers automatically, and the answer is either right or wrong.

There are only two kinds of question, picked 50/50:

- **HARD**: it needs two searches to be sure of the answer.
- **EASY**: it already knows the answer, so every search is wasted money.

**k** is the number of searches in an attempt. Three turns with two choices
each give 2³ = 8 possible attempts for each question type:

```
k = 0:  SKIP SKIP SKIP
k = 1:  SEARCH SKIP SKIP    SKIP SEARCH SKIP    SKIP SKIP SEARCH
k = 2:  SEARCH SEARCH SKIP  SEARCH SKIP SEARCH  SKIP SEARCH SEARCH
k = 3:  SEARCH SEARCH SEARCH
```

SKIP is not a result. It is just the other choice at a turn: SKIP, SKIP, SKIP
means "never searched", k = 0.

**The best plan:** on HARD, search exactly twice; on EASY, never search.
Section 2 shows why, straight from the rewards.

---

## 2. The reward: what each attempt is paid

We defined the reward ourselves, in the environment code. It has two parts.

### 2.1 The rule

1. **Each SEARCH costs 0.1.** The turn that searches is paid −0.1.
2. **After turn 2 the answer is checked.** Right adds **+1** to turn 2's
   reward, and wrong adds **−1**.

The chance of being right depends only on the question type and k:

| p(right) | k = 0 | k = 1 | k = 2 | k = 3 |
|---|---|---|---|---|
| HARD | 0% | 50% | 100% | 100% |
| EASY | 100% | 95% | 90% | 85% |

- **HARD:** with no search it is always wrong. One search is a coin flip. Two
  or more make it certain.
- **EASY:** it is right without searching. Each search lowers the chance a
  little, as if looking things up only confused it.

In code (simplified from `SimpleGRPO/group_env.py:69`–`72`):

```python
for turn in range(3):
    ...                                                   # pick an action (section 3)
    rewards[:, turn] -= 0.1 * action                      # -0.1 if this turn searched
    searches = searches + action                          # k so far
correct = torch.rand(n) < P_CORRECT[qtype, searches]      # right with chance p(right)
rewards[:, -1] += 2 * correct - 1                         # +1 if right, -1 if wrong, on turn 2
```

### 2.2 Every outcome, turn by turn

There are 16 attempts (2 types × 8), and each ends right or wrong. Below is
**every outcome that can actually happen**, with what each turn pays.
Outcomes with a 0% chance are left out.

- **r₀, r₁:** −0.1 if that turn searched, else 0.
- **r₂:** turn 2's own search cost, plus the answer: +1 if right, −1 if wrong.
- **Total** = r₀ + r₁ + r₂.
- **Chance:** how often that outcome happens for that attempt.

**HARD**

| # | Turn 0 | Turn 1 | Turn 2 | k | Answer | Chance | r₀ | r₁ | r₂ = cost + answer | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | SKIP | SKIP | SKIP | 0 | wrong | 100% | 0.0 | 0.0 | 0.0 + (−1.0) = −1.0 | **−1.0** |
| 2 | SEARCH | SKIP | SKIP | 1 | right | 50% | −0.1 | 0.0 | 0.0 + 1.0 = 1.0 | **0.9** |
| | | | | | wrong | 50% | −0.1 | 0.0 | 0.0 + (−1.0) = −1.0 | **−1.1** |
| 3 | SKIP | SEARCH | SKIP | 1 | right | 50% | 0.0 | −0.1 | 0.0 + 1.0 = 1.0 | **0.9** |
| | | | | | wrong | 50% | 0.0 | −0.1 | 0.0 + (−1.0) = −1.0 | **−1.1** |
| 4 | SKIP | SKIP | SEARCH | 1 | right | 50% | 0.0 | 0.0 | −0.1 + 1.0 = 0.9 | **0.9** |
| | | | | | wrong | 50% | 0.0 | 0.0 | −0.1 + (−1.0) = −1.1 | **−1.1** |
| 5 | SEARCH | SEARCH | SKIP | 2 | right | 100% | −0.1 | −0.1 | 0.0 + 1.0 = 1.0 | **0.8** |
| 6 | SEARCH | SKIP | SEARCH | 2 | right | 100% | −0.1 | 0.0 | −0.1 + 1.0 = 0.9 | **0.8** |
| 7 | SKIP | SEARCH | SEARCH | 2 | right | 100% | 0.0 | −0.1 | −0.1 + 1.0 = 0.9 | **0.8** |
| 8 | SEARCH | SEARCH | SEARCH | 3 | right | 100% | −0.1 | −0.1 | −0.1 + 1.0 = 0.9 | **0.7** |

**EASY**

| # | Turn 0 | Turn 1 | Turn 2 | k | Answer | Chance | r₀ | r₁ | r₂ = cost + answer | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | SKIP | SKIP | SKIP | 0 | right | 100% | 0.0 | 0.0 | 0.0 + 1.0 = 1.0 | **1.0** |
| 2 | SEARCH | SKIP | SKIP | 1 | right | 95% | −0.1 | 0.0 | 0.0 + 1.0 = 1.0 | **0.9** |
| | | | | | wrong | 5% | −0.1 | 0.0 | 0.0 + (−1.0) = −1.0 | **−1.1** |
| 3 | SKIP | SEARCH | SKIP | 1 | right | 95% | 0.0 | −0.1 | 0.0 + 1.0 = 1.0 | **0.9** |
| | | | | | wrong | 5% | 0.0 | −0.1 | 0.0 + (−1.0) = −1.0 | **−1.1** |
| 4 | SKIP | SKIP | SEARCH | 1 | right | 95% | 0.0 | 0.0 | −0.1 + 1.0 = 0.9 | **0.9** |
| | | | | | wrong | 5% | 0.0 | 0.0 | −0.1 + (−1.0) = −1.1 | **−1.1** |
| 5 | SEARCH | SEARCH | SKIP | 2 | right | 90% | −0.1 | −0.1 | 0.0 + 1.0 = 1.0 | **0.8** |
| | | | | | wrong | 10% | −0.1 | −0.1 | 0.0 + (−1.0) = −1.0 | **−1.2** |
| 6 | SEARCH | SKIP | SEARCH | 2 | right | 90% | −0.1 | 0.0 | −0.1 + 1.0 = 0.9 | **0.8** |
| | | | | | wrong | 10% | −0.1 | 0.0 | −0.1 + (−1.0) = −1.1 | **−1.2** |
| 7 | SKIP | SEARCH | SEARCH | 2 | right | 90% | 0.0 | −0.1 | −0.1 + 1.0 = 0.9 | **0.8** |
| | | | | | wrong | 10% | 0.0 | −0.1 | −0.1 + (−1.0) = −1.1 | **−1.2** |
| 8 | SEARCH | SEARCH | SEARCH | 3 | right | 85% | −0.1 | −0.1 | −0.1 + 1.0 = 0.9 | **0.7** |
| | | | | | wrong | 15% | −0.1 | −0.1 | −0.1 + (−1.0) = −1.1 | **−1.3** |

Three lines to read slowly:

- **HARD #2, one search on turn 0.** If right (50%), it is paid −0.1, 0.0,
  1.0, a total of **0.9**. If wrong (50%), it is paid −0.1, 0.0, −1.0, a total
  of **−1.1**. The same attempt can score 0.9 or −1.1: that is the **noise**.
- **HARD #5 to #7, two searches.** Always right, and always a total of **0.8**,
  whichever turns searched. The costs land on different turns, but the total
  depends only on k.
- **HARD #8, three searches.** Still right, but the third search cost 0.1 and
  bought nothing: **0.7 < 0.8**.

### 2.3 The average of each attempt

Average the outcomes by their chances, and only the question type and k are
left:

| Average total | k = 0 | k = 1 | k = 2 | k = 3 |
|---|---|---|---|---|
| HARD | 100% × (−1.0) = **−1.0** | 50% × 0.9 + 50% × (−1.1) = **−0.1** | 100% × 0.8 = **0.8** | 100% × 0.7 = **0.7** |
| EASY | 100% × 1.0 = **1.0** | 95% × 0.9 + 5% × (−1.1) = **0.8** | 90% × 0.8 + 10% × (−1.2) = **0.6** | 85% × 0.7 + 15% × (−1.3) = **0.4** |

This is the table that defines the game. The best plan is the highest number
in each row: **two searches on HARD (0.8), none on EASY (1.0)**.

**J** scores a whole policy on this table. Picture the policy playing forever,
on a 50/50 mix of HARD and EASY questions: J is its average total per attempt.
The code computes it exactly, by listing all 16 attempts with how likely the
policy is to make each one (`true_reward`, in `SimplePPO/env.py`). Two
reference points:
- **the best plan:** ½ · 0.8 + ½ · 1.0 = **0.90**;
- **the starting policy:** **0.388**.

Training never sees J. It only ever sees single, noisy attempts.

### 2.4 Same averages, three kinds of noise

The three modules use the same average table, but pay it out differently:

| Module | What turn 2 adds for the answer |
|---|---|
| **SimpleGRPO** | +1 if right, −1 if wrong: the tables above, exactly |
| **SimplePPO** | no right or wrong. It adds the answer's *quality*, the average of +1/−1, which is 2 · p(right) − 1 (HARD −1, 0, 1, 1; EASY 1, 0.9, 0.8, 0.7), plus random Gaussian noise of size about 1. See `SimplePPO/env.py:104`–`106`. |
| **SimpleDPO** | **nothing reaches training.** Each pair is labelled by which attempt has the higher *average* total (section 7). |

The noise is different but the averages are the same. So the three modules
share the same best plan and the same J, and their results can be compared.

---

## 3. How the agent decides: the state and the policy table

### 3.1 The 12 states

Before each decision the agent knows three things:

1. **the question type** (HARD or EASY);
2. **which turn it is** (0, 1 or 2);
3. **how many times it has searched so far.**

Those three facts together are the **state**. At turn t it can have searched
0 to t times, so there are 1 + 2 + 3 = 6 states per question type, 12 in all.
Each one has a number, its **state id**:

```
             turn 0     turn 1          turn 2
so far:        0       0      1       0      1      2
HARD (+0)      0       1      2       3      4      5
EASY (+6)      6       7      8       9     10     11
```

The agent itself, the **policy**, is a table with one row per state and two
columns, SKIP and SEARCH. Each row holds two scores, the **logits**. Softmax
turns them into the probabilities of SKIP and SEARCH in that state. The
state id is simply **which row to read**:

```python
policy.logits[state]      # the [SKIP, SEARCH] logits for this state
```

Every run starts from the same table: p(SEARCH) = 0.4 and p(SKIP) = 0.6 in
every row (`SimplePPO/env.py:67`). Learning means changing these 24 numbers.

**"HARD turn 0 is 0 and EASY turn 0 is 6"** means they are different rows.
The same moment, turn 0 with nothing searched yet, gets its own row for each
question type, and so its own p(SEARCH).

### 3.2 How the code numbers the states

```python
def state_id(qtype, turn, searches):
    return qtype * 6 + turn * (turn + 1) // 2 + searches
```

It adds three offsets. Take EASY, turn 2, one search so far:

1. **`qtype * 6` picks the half of the table.** HARD is rows 0–5 and EASY is
   rows 6–11. For EASY: 1 × 6 = **6**.
2. **`turn * (turn + 1) // 2` skips the earlier turns' rows.** Turn 0 has 1
   row and turn 1 has 2, so turn 1 starts at offset 1 and turn 2 at
   1 + 2 = 3. For turn 2: **3**.
3. **`searches` picks the row within the turn.** One search so far: **1**.

Total: 6 + 3 + 1 = **10**.

There is no row for "3 searches so far". That can only happen after the last
turn, when the agent answers automatically and has nothing left to decide.

### 3.3 One attempt through the table

Take SEARCH, SEARCH, SKIP (attempt #5 in section 2):

| Turn | Searches so far | Row on HARD | Row on EASY | Action | Reward on HARD |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 6 | SEARCH | −0.1 |
| 1 | 1 | 2 | 8 | SEARCH | −0.1 |
| 2 | 2 | 5 | 11 | SKIP | 0.0 + 1.0 (always right) = 1.0 |

On HARD it reads rows 0, 2, 5 and scores 0.8. On EASY it reads rows 6, 8, 11:
the same actions, but different rows. In code, `rollout` does this for the
whole batch at once:

```python
state = state_id(qtype, turn, searches)     # one row per attempt
action = policy.dist(state).sample()        # draw SKIP or SEARCH from that row's probabilities
searches = searches + action                # update "searches so far" for the next turn
```

So **the state decides the action, and the reward depends on the actions
taken, plus the coin for right or wrong.** A state does not "have" a reward.
But it does decide what reward is still possible from there. PPO's critic
learns exactly that (section 5).

### 3.4 Why the question type has to be in the state

Suppose the state were only (turn, searches so far). Then HARD and EASY would
share rows, and turn 0 would have a single p(SEARCH). It would need to be high
for HARD and low for EASY at the same time, and no single number can be both.

With separate rows, training sets them independently. A trained SimpleDPO
policy (seed 0) ends with p(SEARCH) = **1.00 in row 0** (HARD, turn 0) and
**0.00 in row 6** (EASY, turn 0), rounded.

"Searches so far" is in the state for the same reason. On HARD turn 2, the
agent should SEARCH if it has searched once (row 4), but SKIP if it has
searched twice (row 5).

---

## 4. The problem all three algorithms solve

Every algorithm has to decide which actions to make more likely, from
attempts like the ones in section 2. Two things make that hard:

1. **The noise.** The same attempt can score 0.9 or −1.1 (HARD #2). One
   attempt says little; only many attempts reveal the average.
2. **Fairness across question types.** An EASY attempt scoring 0.8 looks
   better than a HARD attempt scoring 0.7. But 0.8 on EASY means "wasted a
   search" (the best is 1.0), while 0.7 on HARD is close to the best (0.8).

So each algorithm judges an attempt against a **baseline**: what a typical
attempt at *the same kind of question* scores. The three modules differ
mainly in where that baseline comes from:

| | The yardstick | Why it never mixes up HARD and EASY |
|---|---|---|
| **SimplePPO** | the critic: a learned guess for each of the 12 states | the state includes the question type |
| **SimpleGRPO** | the average of the other attempts at the same question | all of them answer the same question type |
| **SimpleDPO** | the other attempt in the pair | both answer the same question type, so it cancels out |

They also get the same budget: **16 attempts per iteration**, laid out
differently:

```
SimplePPO    16 questions x 1 attempt     q0 q1 q2 q3 ... q15             each question tried once
SimpleGRPO    2 questions x 8 attempts    q0 q0 q0 q0 q0 q0 q0 q0 | q1 q1 q1 q1 q1 q1 q1 q1
SimpleDPO     8 questions x 2 attempts    q0 q0 | q1 q1 | q2 q2 | ... | q7 q7
```

---

## 5. SimplePPO: one attempt per question, a critic as the yardstick

**What it collects:** 16 questions, each tried once, from the current policy.
Rewards come from section 2.4's SimplePPO row: −0.1 per search, plus the
answer's quality and Gaussian noise on turn 2 (`SimplePPO/env.py`, `rollout`).

A real batch of 4 (`env.rollout(policy, critic, 4)`, seed 0, starting policy):

```
qtype     [HARD, EASY, EASY, HARD]        4 different questions, each tried once
actions   [[SKIP,   SEARCH, SKIP],        k = 1
           [SKIP,   SKIP,   SKIP],        k = 0
           [SKIP,   SKIP,   SKIP],        k = 0
           [SEARCH, SEARCH, SKIP]]        k = 2
states    [[0, 1, 4], [6, 7, 9], [6, 7, 9], [0, 2, 5]]
rewards   [[ 0.00, -0.10, -1.29],
           [ 0.00,  0.00,  0.21],
           [ 0.00,  0.00,  0.98],
           [-0.10, -0.10,  0.28]]
```

- **Attempt 0 (HARD, one search on turn 1):** it pays −0.1 on turn 1. Turn 2
  adds one search's quality on HARD, 0.0, plus noise of −1.29.
- **Attempt 3 (HARD, two searches: the best plan):** it still got only
  1.0 − 0.72 = 0.28 on turn 2.

**The noise is as big as the signal.** One attempt tells PPO very little.

**The critic** is PPO's yardstick: a second table, of 12 numbers, one per
state. Each is a guess of the reward **still to come** from that state.
- **HARD, turn 2, two searches so far (row 5):** under a policy that skips
  there, the guess should be about 1.0, because the answer is already as good
  as it gets.
- **Row 0:** the average of everything the policy might still do on a HARD
  question.

The critic learns these guesses from the rewards it sees. After 60 iterations
(seed 0), it guesses 0.51 for row 0 (HARD) and 1.12 for row 6 (EASY), against
true values of 0.76 and 0.97. That is rough, but it rates EASY above HARD,
which is what a yardstick needs.

**GAE** then turns rewards and guesses into an **advantage for each turn**:
did things go better or worse than the critic expected, from this turn on?
At the very start the critic knows nothing (all 12 guesses are 0), so the
advantage is just the reward still to come, each later turn weighted by
λ = 0.8. For attempt 0, working backwards from the last turn:

```
A₂ = -1.29
A₁ = -0.10 + 0.8 × (-1.29) = -1.13
A₀ =  0.00 + 0.8 × (-1.13) = -0.91        advantages [-0.91, -1.13, -1.29]
```

All three are negative, so the update makes each of those three actions less
likely, in its own state. As the critic improves, the guess for each state is
subtracted, and an action gets a positive advantage only when it beat the
critic's guess for that state. PPO never compares two attempts directly: every
comparison goes through the critic.

---

## 6. SimpleGRPO: eight attempts per question, the group as the yardstick

**What it collects:** 2 questions, each tried 8 times, from the current policy.
Rewards are exactly section 2.2's: −0.1 per search, ±1 for the answer
(`SimpleGRPO/group_env.py`, `rollout`).

A real group (`group_env.rollout`, seed 3, shortened to 4 attempts so it fits;
the module uses 8). The question happened to be HARD:

```
attempt   actions                 k   answer   total
   1      SKIP   SKIP   SKIP      0   wrong    -1.0
   2      SKIP   SKIP   SEARCH    1   right     0.9      one search: a coin flip, won
   3      SEARCH SEARCH SKIP      2   right     0.8
   4      SKIP   SKIP   SEARCH    1   wrong    -1.1      one search: a coin flip, lost

group average -0.1, spread (standard deviation) 1.10
advantage = (total - group average) / spread  =  [-0.82, 0.91, 0.82, -0.91]
```

Each attempt gets **one** advantage, copied to all three of its turns. There
is no critic. "Is 0.8 good?" is answered by the other attempts at the same
question, so the question type takes care of itself.

Notice attempt 2: one lucky search scored more than the correct plan of two
searches, and got the bigger push. Over many groups that luck averages out,
because one search loses half the time (attempt 4).

---

## 7. SimpleDPO: two attempts per question, labelled by which is better

**What it collects:** 8 questions, each tried twice, by the **starting policy
π_ref**. Everything it will train on is collected and labelled once, before
training starts. This is called *offline*, and it is how the published DPO
works.

**What it gets back: no reward at all.** Only a **label** for each pair: which
of the two attempts is better. The better one is **chosen** and the other is
**rejected**.

### 7.1 Where the labels come from

**DPO itself never asks anyone anything.** It starts from a dataset where
every pair is already labelled (chosen, rejected), and it only learns from
those labels. A real DPO project has two separate steps:

```
STEP 1, before DPO: make the dataset
    for each prompt:
        the starting model (π_ref) writes two answers
        a labeller picks the better one               <- people, a strong AI judge, or a checker
    save (prompt, chosen, rejected)

STEP 2, DPO: train on that fixed dataset
    make chosen more likely and rejected less likely, compared with π_ref
```

The toy has no people, so it does step 1 itself, with the simplest rule:

- **the attempt with the higher average total (section 2.3) is chosen**, the
  other is rejected;
- **two equally good attempts are skipped.** Same question type and same k
  means the same average, so the pair has nothing to teach.

The code keeps the two steps apart:

| | Step 1: make the labelled pairs | Step 2: DPO |
|---|---|---|
| File | `SimpleDPO/pair_env.py`: given code | `SimpleDPO/simple_dpo.py`: the algorithm you build |
| Function | `collect_pairs`: π_ref makes 2 attempts, the better one is chosen | `dpo_loss`, `dpo_update` |
| Sees | the attempts' average totals | only (chosen, rejected) and π_ref's chances of each |
| When | all of it, before training starts (`pair_env.py:114`) | the 60 iterations of training (`pair_env.py:122`) |

So by the time `dpo_update` runs, the labels are fixed data. The DPO paper does
the same in one of its experiments: it had no human labels for movie-review
sentiment, so a sentiment classifier picked the chosen review of each pair.

### 7.2 A and B: the two attempts, before labelling

**A** and **B** are just names for the two attempts at the same question, in
the order the policy made them: **A is the first attempt, B is the second.**
Labelling turns them into **chosen** and **rejected**:

```
  A (first attempt)  ─┐  compare their     ┌─►  chosen    = the better one
                      ├───────────────────►┤
  B (second attempt) ─┘  average totals    └─►  rejected  = the other one
```

So chosen can be A or B, depending on which one is better.

One pair, step by step:

1. **A question is drawn.** Say HARD.
2. **π_ref makes two attempts at it.**
   - A: SEARCH, SEARCH, SKIP (k = 2, average total 0.8).
   - B: SKIP, SKIP, SKIP (k = 0, average total −1.0).
3. **Compare the averages:** 0.8 > −1.0, so chosen = A and rejected = B.
4. **Only (chosen, rejected) is stored.** The averages, 0.8 and −1.0, are
   thrown away. DPO never sees them.

In code (simplified from `SimpleDPO/pair_env.py:70`–`81`), attempts 2i and
2i + 1 answer question i, so A is column 0 and B is column 1 of each pair:

```python
batch = rollout(policy, ref_policy, questions, 2)       # 2 attempts per question: A, B, A, B, ...
scores = QUALITY[qtype, searches].view(-1, 2)           # average totals: one row per question, [A, B]
first_wins = scores[:, 0] > scores[:, 1]                # is A better?
keep = (scores[:, 0] - scores[:, 1]).abs() > 1e-6       # equally good: skip the pair
chosen   = (firsts + (~first_wins))[keep]               # A's index if A is better, else B's
rejected = (firsts + first_wins)[keep]                  # the other one
```

(`firsts` is 0, 2, 4, …, the index of each question's A.)

### 7.3 Which attempt is chosen: every case

Rank each question type's search counts by average total (section 2.3):

```
HARD:  k = 2 (0.8)  >  k = 3 (0.7)  >  k = 1 (−0.1)  >  k = 0 (−1.0)
EASY:  k = 0 (1.0)  >  k = 1 (0.8)  >  k = 2 (0.6)   >  k = 3 (0.4)
```

For any pair, the attempt that comes first in this order is chosen. If both
have the same k, the pair is skipped. Three cases worth reading slowly:

- **HARD, two searches against three:** two searches is chosen (0.8 > 0.7).
  Both are always right; the third search only cost 0.1.
- **HARD, one search against none:** one search is chosen (−0.1 > −1.0), even
  though it is right only half the time. The label compares averages, not
  what happened this time.
- **EASY, no search against one:** no search is chosen (1.0 > 0.8).

About a third of all pairs are ties and get skipped: of the 480 questions in a
run, about 324 give a labelled pair.

### 7.4 A real batch

`collect_pairs`, 4 questions, seed 3, from π_ref:

| Question | Type | A (average total) | B (average total) | Label |
|---|---|---|---|---|
| 0 | HARD | SEARCH SEARCH SKIP (0.8) | SEARCH SKIP SKIP (−0.1) | chosen = A |
| 1 | HARD | SKIP SKIP SKIP (−1.0) | SKIP SEARCH SEARCH (0.8) | chosen = B |
| 2 | EASY | SKIP SKIP SKIP (1.0) | SKIP SKIP SKIP (1.0) | equally good: skipped |
| 3 | EASY | SEARCH SEARCH SKIP (0.6) | SKIP SKIP SEARCH (0.8) | chosen = B |

Four questions give three labelled pairs. In question 0 the better attempt
came first (A), in questions 1 and 3 second (B). Question 2's two attempts
were identical, so there was nothing to learn from them.

### 7.5 What DPO does with a pair

The training data is only (chosen, rejected), plus how likely π_ref was to
produce each one. For an attempt with k searches, that is 0.4^k · 0.6^(3−k).
For example, two searches: 0.4 × 0.4 × 0.6 = 0.096.

DPO's loss makes the chosen attempt **more likely** and the rejected one
**less likely**, measured against those π_ref chances. Both attempts answer
the same question, so the question type cancels out: the loss never needs to
know how hard the question was. The policy's HARD and EASY rows stay separate
(section 3), so a HARD pair only moves HARD rows.

The loss itself, worked through with numbers for one pair, is in
`./scripts/run_simple_dpo.sh steps` (steps 3 to 6).

After training, look at how much more likely the policy has made each attempt
than π_ref did, on a log scale and times β: β · log(π / π_ref). Averaged over
20 seeds, relative to k = 0:
- **HARD:** [0, 2.44, 4.53, 4.42];
- **EASY:** [0, −1.31, −3.48, −4.39].

That is **the same order as the average totals** (HARD [0, 0.9, 1.8, 1.7], EASY
[0, −0.2, −0.4, −0.6]), which DPO never saw. Trained only on which attempt was
better, the policy has become a scorer of attempts. Its numbers are bigger
than the averages, and keep growing with training, because labels that never
disagree never tell it to stop.

### 7.6 Two things to keep in mind

- **"Chosen" does not mean "correct".** Two searches are chosen over three on
  HARD although both are always right. One search is chosen over none
  although it is wrong half the time. Chosen only means "the better of these
  two, on average".
- **The labels know the true averages.** That is information PPO and GRPO
  never get: they only see what each attempt actually scored. SimpleDPO's
  `labels="outcome"` option makes the comparison fair, by labelling with the
  totals the attempts actually got (the right/wrong lines in section 2.2), the
  way verl's online-DPO recipe labels with a checker. On HARD, one search
  against two then comes down to the one-search attempt's coin: 0.9 beats 0.8
  when it is right, and −1.1 loses when it is wrong. It is 50/50, although two
  searches are far better on average. That is why the demo's "by outcome,
  online" row gets stuck.

---

## 8. How the experiments run

Every `run_*.py` follows the same plan:

```
for each variant (one row of the results table):
    for seed in 0..19:
        policy = the starting policy (p(SEARCH) = 0.4 in every row)
        for iteration in 1..60:
            collect 16 attempts            PPO: 16 x 1,  GRPO: 2 x 8,  DPO: 8 x 2, labelled into pairs
            compute the learning signal    per-turn advantages / per-attempt advantages / pairs
            update the policy              10 passes over the batch, in small slices (SGD)
            J = true_reward(policy)        exact; recorded, never used for training
    report the average J after 10, 30 and 60 iterations
    (SimpleGRPO and SimpleDPO also count "stuck" seeds: J still below 0.7 at the end)
```

The default row of each module, after 60 iterations (best possible 0.90):

| Module | Default run | J after 60 |
|---|---|---|
| SimplePPO | critic + GAE + clip, Gaussian-noise rewards | 0.854 |
| SimpleGRPO | groups of 8, ±1 rewards | 0.839 |
| SimpleDPO | about 324 labelled pairs from π_ref, β 0.1 | 0.899 |

These are not a race. PPO and GRPO see noisy rewards: a Gaussian, a coin flip.
DPO's labels are made from the true averages, which is why it nearly reaches
the best possible J. To compare across modules on equal terms:
- SimpleGRPO's table has a PPO row on GRPO's ±1 rewards (0.869);
- SimpleDPO's table labels pairs by the rewards they actually got ("by
  outcome"), and has two GRPO rows on DPO's budget of 8 × 2 attempts.

---

## 9. From the toy to an LLM

| The toy | An LLM |
|---|---|
| question type (2 kinds) | the prompt (every one is its own kind) |
| an attempt | a response |
| SEARCH or SKIP at a turn | the next token |
| state: (type, turn, searches so far) | the prompt plus the tokens so far |
| the average-total table | a reward model, or a checker that says right or wrong |
| SimpleDPO's labeller | people, or a judge model, picking the better of two responses |

The three yardsticks matter much more on an LLM:
- **PPO's critic** must guess how good every *prompt* is. It is a second
  network, as large as the policy.
- **GRPO** samples several responses to *the same prompt*, so their average is
  a yardstick for that prompt, for free.
- **DPO** compares two responses to *the same prompt*, so the prompt's
  difficulty cancels out.

On the toy there are only two question types, and the policy can see which one
it has. So even one shared yardstick for the whole batch works nearly as well:
SimplePPO's "no critic" row scores 0.845, and SimpleGRPO's "batch mean" row
0.833.
