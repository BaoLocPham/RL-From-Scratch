# Build TRPO from scratch

Do not open `trpo.py` (the reference) before you finish the exercise. Work from
the docstrings in `from_scratch/trpo.py` and the grader's messages; reading the
reference turns this into transcription.

This follows the PPO paper's Section 2.2 (eq. 3-4). Finish `VPG/` first: the
toy, the policy and the rollout collector are VPG's, imported from
`VPG/vpg.py`, and stage 3's optimizer lines are the ones you wrote there.

| Stage | Function | Question it answers |
|---|---|---|
| 1 | `surrogate` | Why does eq. 3 have the same gradient as L^PG at θ_old, but a different value after? |
| 2 | `mean_kl` | Why does KL use both actions, and why is `old_probs` the one doing the weighting? |
| 3 | `trpo_update` | Why snapshot θ_old once, but the logits every epoch? |
| 4 | none | What does the KL fence save on the batch that broke VPG? |

## How to start

Each stage is two or three TODO lines in `from_scratch/trpo.py`. Stages 1 and 2
are the two equations; stage 3's loop and optimizer calls are written for you,
and only the fence around them is left blank.

1. Open `from_scratch/trpo.py` and fill the lines marked `TODO stage 1`.
2. Try them: `python TRPO/from_scratch/trpo.py` prints your result for every
   stage next to the expected one; unfinished stages say "not done yet".
3. Check them: `./scripts/run_trpo.sh check` grades the stages in order and stops
   at the first one that is not right yet, with a hint about the likely mistake.
4. Repeat for stages 2 and 3. When all pass, `./scripts/run_trpo.sh diff` proves
   your walkthrough output matches the reference line for line.

Stage 4 needs no new code: it runs your `trpo_update` on the unlucky seed-17
batch. VPG's 100 epochs on it dropped J from 0.600 to 0.373; yours should stop
after 6 epochs, at 0.559.

## The running example

The same batch as VPG's: 10 rollouts on HARD questions, collected at
p(tool) = 0.40. 4 called the tool and went well (Â = +3); 6 answered directly
and went badly (Â = −2).

Stage 3 runs it with lr 0.02, small enough that several epochs fit inside the
trust region. KL after each epoch: 0.0003, 0.0011, 0.0026, 0.0046, 0.0072, then
0.0105 on the sixth, which is over δ = 0.01, so the sixth is undone and 5 are
kept. p(tool|HARD) ends at 0.460, between the 0.45 (allowed) and 0.48 (not
allowed) moves worked out on the Notion page.

## Terms in the TRPO logs

The VPG terms (p(tool|HARD), true J, rollout, batch, update, epoch, iteration)
are in `VPG/README.md`. TRPO adds:

- **ratio**: π_θ(a|s) / π_θ_old(a|s), how much more (or less) likely an action is
  now than when the batch was collected. 1 = unchanged.
- **surrogate**: eq. 3, mean(ratio × Â). Like L^PG, but it knows which model
  collected the batch.
- **KL**: how much the policy's *behaviour* has changed since θ_old, over both
  actions. 0 = unchanged. Computed exactly here; an LLM has to estimate it (see
  *From this toy to an LLM* in the root README).
- **KL constraint**: eq. 4, KL ≤ δ, the limit on how far one iteration may move
  the policy.
- **δ** (`max_kl`): the most KL one iteration may use. Here 0.01, in nats.
- **trust region**: every policy within KL ≤ δ of θ_old, where the batch is
  still trusted.

## A simplification: how the constraint is enforced

The paper's TRPO finds the best step inside the trust region in one go:
it linearizes eq. 3, approximates eq. 4 quadratically, and solves with conjugate
gradient. This exercise takes small SGD steps on eq. 3 and rolls back the first
one that leaves the trust region. Both respect eq. 3 and eq. 4; the rollback is
cruder but first-order, and it leaves TRPO's flaw in plain sight: the
constraint is checked *outside* the loss, after every step. PPO moves it into
the loss with a clip.

## At the end, you should be able to answer

- In stage 1, the surrogate is 0 at θ_old and its gradient equals L^PG's. What
  number did L^PG give there, and why does its value not matter either?
- In stage 2, KL(old ‖ new) and KL(new ‖ old) differ. Which distribution
  generated the batch, and why should that one do the weighting?
- In stage 3, what happens if `old_probs` is taken inside the loop, and why
  does the grader see all 50 epochs kept?
- Why must `before` be `.detach().clone()` and not just `.detach()`?
- In stage 4, TRPO still moved J the wrong way (0.600 → 0.559). What limits the
  damage, and what actually repairs it?

## Next

`Surrogates/` puts TRPO beside the other candidates for the same loop: no
limit at all, a fixed or adaptive KL penalty, and the clip. TRPO is the only
one that needs to change the loop itself, not just the loss.
