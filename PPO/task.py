"""The token task: PPO at verl's shapes, small enough to follow by hand. Given code.

The toy track had ONE decision per question. An LLM makes one decision per
token, and the reward arrives only at the end. This task is the smallest thing
with that shape:

  * A response is up to 4 tokens from a vocabulary of 3. The "model" is a table
    of logits, one row per position -- a stand-in for a transformer, where the
    state at position t would be the whole prefix.
  * Half the batch is 4 tokens long, half is 3, so `response_mask` matters.
  * The reward is the fraction of positions that match a hidden target
    (2, 0, 1, 2), and it lands on the LAST real token only. Which tokens earned
    it? That is the question the critic and GAE (stage 2) answer.
  * A value head, one number per position, is the critic V(s_t).

Every tensor that leaves `rollout` is (batch, response_length), exactly what a
verl trainer passes around.
"""

import torch

RESPONSE_LENGTH = 4
VOCAB = 3
TARGET = torch.tensor([2, 0, 1, 2])          # the hidden answer, never shown to the model
LENGTHS = torch.tensor([4, 3])               # two response lengths, alternating in the batch


class TokenModel:
    """The actor (policy logits per position), the critic (a value per position), and a frozen reference."""

    def __init__(self):
        self.policy_logits = torch.zeros(RESPONSE_LENGTH, VOCAB, requires_grad=True)   # theta: uniform at start
        self.value_head = torch.zeros(RESPONSE_LENGTH, requires_grad=True)             # V(s_t)
        self.reference_logits = torch.zeros(RESPONSE_LENGTH, VOCAB)                     # frozen, for the KL

    def parameters(self):
        return [self.policy_logits, self.value_head]

    def logits(self, n):
        """The "forward pass": (n, response_length, vocab) logits."""
        return self.policy_logits.expand(n, -1, -1)

    def values(self, n):
        """The critic's prediction for every position: (n, response_length)."""
        return self.value_head.expand(n, -1)


def response_mask_for(n):
    """Rows alternate between the two lengths: [1, 1, 1, 1] and [1, 1, 1, 0]."""
    lengths = LENGTHS.repeat(n // 2)
    return (torch.arange(RESPONSE_LENGTH)[None, :] < lengths[:, None]).float(), lengths


@torch.no_grad()
def rollout(model, n, logprobs_from_logits):
    """Sample n responses from the model as it is NOW (theta_old) and score them.

    Returns a dict of (n, response_length) tensors, plus `correct`, the reward
    per response. `logprobs_from_logits` is passed in so the rollout uses yours.
    """
    response_mask, lengths = response_mask_for(n)
    probs = model.logits(n).softmax(-1)
    tokens = torch.multinomial(probs.reshape(-1, VOCAB), 1).reshape(n, RESPONSE_LENGTH)

    # Outcome reward: the fraction of real positions that matched, parked on the
    # last real token -- the way an outcome-rewarded verl batch carries it.
    correct = ((tokens == TARGET) * response_mask).sum(-1) / lengths
    token_level_scores = torch.zeros(n, RESPONSE_LENGTH)
    token_level_scores[torch.arange(n), lengths - 1] = correct

    reference = model.reference_logits.expand(n, -1, -1)
    return {
        "tokens": tokens,
        "response_mask": response_mask,
        "token_level_scores": token_level_scores,
        "old_log_prob": logprobs_from_logits(model.logits(n), tokens),     # log pi_theta_old, frozen
        "ref_log_prob": logprobs_from_logits(reference, tokens),           # log pi_ref, for the KL
        "values": model.values(n) * response_mask,                         # V at collection time
        "correct": correct,
    }


def train(impl, verl=False, iterations=40, batch_size=32, lr=0.02, seed=0, every=None):
    """Algorithm 1, outer loop: collect -> compute_advantage -> ppo_update, `iterations` times.

    `impl` is a module: the reference (PPO/common.py) or yours
    (PPO/from_scratch/ppo.py, or ppo_verl.py for verl=True). verl=False runs core
    PPO (compute_advantage, ppo_update); verl=True runs the same loop with verl's
    functions (verl_compute_advantage, verl_ppo_update). Returns one row of metrics per
    iteration, and the model.
    """
    advantage = impl.verl_compute_advantage if verl else impl.compute_advantage
    update = impl.verl_ppo_update if verl else impl.ppo_update
    torch.manual_seed(seed)
    model = TokenModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    if every:
        print(header(verl))
    for iteration in range(iterations):
        batch = rollout(model, batch_size, impl.logprobs_from_logits)       # theta_old
        batch = advantage(batch)                                            # rewards -> advantages, once
        metrics = update(model, optimizer, batch)                           # K epochs of minibatches
        metrics["reward"] = float(batch["correct"].mean())
        history.append(metrics)
        if every and (iteration % every == 0 or iteration == iterations - 1):
            print(row(iteration, metrics))
    return history, model


COLUMNS = (("reward", 7, ".3f"), ("pg_loss", 9, ".4f"), ("vf_loss", 8, ".4f"), ("entropy", 8, ".4f"),
           ("pg_clipfrac", 9, ".3f"), ("ppo_kl", 8, ".4f"))


def header(verl=False):
    names = COLUMNS if verl else COLUMNS[:4]                           # clipfrac and ppo_kl are verl's metrics
    return f"{'iteration':>9} " + " ".join(f"{name.replace('pg_clip', 'clip'):>{width}}" for name, width, _ in names)


def row(iteration, metrics):
    cells = [f"{metrics[name]:>{width}{fmt}}" for name, width, fmt in COLUMNS if name in metrics]
    return f"{iteration:>9} " + " ".join(cells)
