"""E4: a training file of synthetic addresses plus k real ones.

The k real addresses are the first k of the real-train pool after a shuffle with the seed, so the subsets are nested
within a seed (25 inside 50 inside 100 inside 200). They are repeated until they make up about a tenth of the file:
25 addresses among 20,000 would hardly be seen. This is the proposed rule and is still to be confirmed (initials.md §12).

Run: uv run python -m src.mix --synthetic data/synthetic/full/train.jsonl --real data/real/train.jsonl --k 50 --seed 42 \
         --out data/synthetic/mixed/full-k50-s42/train.jsonl
"""
import argparse
import random

from src.data import read_jsonl, require_train, write_jsonl

SHARE = 0.10  # the part of the mixed file that is real, whatever k is


def mix(synthetic: list[dict], real: list[dict], k: int, seed: int, share: float = SHARE) -> list[dict]:
    if not 0 <= k <= len(real):
        raise ValueError(f"k = {k}, but the real pool holds {len(real)} addresses")
    if k == 0:
        return list(synthetic)
    require_train(real, "the real pool")
    chosen = random.Random(seed).sample(real, len(real))[:k]  # a full shuffle first, so a smaller k is a prefix of a larger one
    copies = max(1, round(share * len(synthetic) / ((1 - share) * k)))
    return list(synthetic) + chosen * copies


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", required=True)
    p.add_argument("--real", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    synthetic = read_jsonl(a.synthetic)
    mixed = mix(synthetic, read_jsonl(a.real), a.k, a.seed)
    write_jsonl(a.out, mixed)
    print(f"{a.out}: {len(synthetic)} synthetic + {len(mixed) - len(synthetic)} real rows ({a.k} addresses)")
