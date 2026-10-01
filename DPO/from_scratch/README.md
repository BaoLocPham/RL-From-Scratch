# Build verl's online DPO from scratch

Do not open `../common.py` first. Work from the docstrings in `dpo.py` and the
grader's messages.

**Do `SimpleDPO/` first** if you have not. It builds the same loss on the toy,
derives it, and shows what β, online pairs and outcome-ranked pairs do. This
exercise is that loss at verl's shapes, the way verl's code writes it.

verl's trainer (`verl/trainer/ppo/core_algos.py`) has no DPO. Its DPO lives in
the **SPIN recipe** ([Self-Play Fine-Tuning](https://arxiv.org/abs/2401.01335)),
now in the separate
[`verl-project/verl-recipe`](https://github.com/verl-project/verl-recipe/tree/main/spin)
repository, as three functions in `spin/core_algos.py`. They are this exercise:

| Stage | Function | Question it answers |
|---|---|---|
| 1 | `compute_onlinedpo_pref` | Two sampled responses per prompt: which one is chosen? |
| 2 | `get_batch_logps` | How do the logits of prompt + response become log π(response)? |
| 3 | `compute_online_dpo_loss` | DPO, label smoothing, IPO, and no reference at all |

Run `python DPO/from_scratch/check.py` from the repository root, or
`./scripts/run_dpo.sh check`. Nothing here needs your PPO or GRPO exercise.

## The loop around them

`spin_trainer.fit_dpo` and `dp_actor.update_policy_dpo_with_ref`, in short:

```
for step:
    generate rollout.n = 2 responses per prompt, adjacent in the batch; score them
    preferences = compute_onlinedpo_pref(token_level_rewards, response_mask)        stage 1
    split into chosen / rejected; labels = input_ids with the prompt set to -100
    reference_*_logps = (ref_log_prob * response_mask).sum(-1)
    loss = compute_online_dpo_loss(get_batch_logps(chosen), get_batch_logps(rejected),   stages 2, 3
                                   reference_chosen_logps, reference_rejected_logps, beta)
    one optimizer step
    every ref_update_freq steps: copy the actor into the reference
```

The splitting and the metrics are given in `../common.py` as `prepare_dpo_batch`
and `dpo_metrics`. The recipe's config sets `dpo_beta: 0.1`, `rollout.n: 2` and
`ref_update_freq: 1`.

## Details that are easy to get wrong

1. **Pairs are adjacent rows, not uids.** GRPO grouped rows by a `uid` array.
   Here rows 2i and 2i + 1 are the two responses to prompt i, so `.view(-1, 2)`
   finds the pairs.
2. **A tie is not dropped.** `torch.argmax` returns the first maximum, so a tied
   pair still becomes (first, second). If the two responses are identical,
   their gradients cancel exactly. If they differ, the first gets pushed up for
   nothing: noise. `SimpleDPO/` drops ties instead.
3. **The shift.** A causal LM's logits at position t predict the token at t + 1:
   `logits[:, :-1]` goes with `labels[:, 1:]`.
4. **-100 is not a token.** It marks positions that are not scored. `gather`
   cannot index with it, so replace it before gathering and mask afterwards.
5. **Sum, not mean.** DPO compares whole responses. `average_log_prob=True`
   exists, and the recipe does not use it.

## One thing the recipe does differently from its own reference

The labels mask only the prompt (`labels[:, :prompt_len] = -100`). Padding
after a response keeps its PAD token id, so `get_batch_logps` counts it in the
policy's sum. The reference's sum, over `response_mask`, never does.
`steps_dpo.py` step 3 prints the difference on a padded row. This repo keeps
the recipe's behaviour in `prepare_dpo_batch` and says so there; its demo uses
equal-length responses, where the two agree.

## The loss options, in one line each

- `loss_type="sigmoid"`: DPO, −log σ(β · logits).
- `label_smoothing=ε`: assume each label is flipped with probability ε
  (conservative DPO). The loss then has a finite best margin.
- `loss_type="ipo"`: IPO, (logits − 1/(2β))². It aims at a fixed margin, so
  a pair that is always decided the same way is not pushed without end.
- `reference_free=True`: `ref_logratios` = 0. Nothing anchors the policy to π_ref.

## At the end, you should be able to answer

- With `ref_update_freq: 1`, the reference is last step's policy. What are the
  logits at the start of every update, and what does the loss reduce to?
- `DPO/run_dpo.py` shows `rewards_chosen` and `rewards_rejected` both rising.
  Why doesn't the loss prevent that, and when is it a problem?
- A tied pair of different responses gets a gradient. How would you change
  `compute_onlinedpo_pref` to drop ties, and what would the batch size become?
- The recipe's `rewards_accuracies` is 0 or 1 per batch. How does it differ from
  the fraction of pairs ranked correctly?
- DPO was derived for pairs judged by a Bradley–Terry rater. The recipe ranks
  by a verifiable reward instead. `SimpleDPO/` shows one way that can fail.
  Which one, and what does GRPO keep that a preference throws away?
