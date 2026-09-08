# Build DPO from scratch

Do not open `../common.py` first. DPO is compact enough that the entire learning
value lies in reconstructing why four log-probabilities become one binary loss.

| Stage | Function | Core idea |
|---|---|---|
| 1 | `sequence_log_probs` | Score a whole completion while ignoring padding |
| 2 | `preference_logit` | Compare policy preference change against a reference |
| 3 | `dpo_loss` | Fit chosen/rejected pairs with a stable logistic objective |
| 4 | `preference_accuracy` | Measure whether pair margins improved |

Run `python DPO/from_scratch/check.py`. The preference logit and loss follow
equations (5-7) in
[Direct Preference Optimization](https://arxiv.org/abs/2305.18290).
The reference implementation names its intermediates `chosen_logratios`,
`rejected_logratios`, and `delta_score` to match
[Hugging Face TRL's standard reverse-KL DPO branch](https://github.com/huggingface/trl/blob/8056842449d2abbc08abd2628c5608071f433bfc/trl/trainer/dpo_trainer.py#L1420-L1458).

At the end, you should be able to explain why token log-probabilities are summed,
why raw chosen likelihood alone is insufficient, what the reference model
anchors, and how `beta` changes the scale of the preference update.
