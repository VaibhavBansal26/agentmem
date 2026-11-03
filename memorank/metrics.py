import random
import re
from collections import Counter


def recall_at_k(ranked_ids, gold_ids, k):
    if not gold_ids:
        return None
    return len(set(ranked_ids[:k]) & set(gold_ids)) / len(set(gold_ids))


def mrr(ranked_ids, gold_ids):
    if not gold_ids:
        return None
    gold = set(gold_ids)
    for i, mid in enumerate(ranked_ids, 1):
        if mid in gold:
            return 1.0 / i
    return 0.0


def _tokens(s):
    return re.findall(r"\w+", s.lower())


def token_f1(pred, gold):
    p, g = _tokens(pred), _tokens(gold)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision, recall = n / len(p), n / len(g)
    return 2 * precision * recall / (precision + recall)


def mean_ci(values, n_boot=1000, seed=0):
    """Mean with a 95% bootstrap CI over questions. None values are dropped."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, None, 0
    mean = sum(vals) / len(vals)
    rng = random.Random(seed)
    boots = sorted(
        sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n_boot)
    )
    return mean, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1], len(vals)
