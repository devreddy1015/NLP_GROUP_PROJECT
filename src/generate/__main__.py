import argparse

from src.generate import CONFIGS, generate

p = argparse.ArgumentParser(description="Generate synthetic addresses. Example: uv run python -m src.generate --config clean --seed 42")
p.add_argument("--config", choices=sorted(CONFIGS), default="clean")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--n-train", type=int, default=20000)
p.add_argument("--n-dev", type=int, default=2000)
p.add_argument("--n-test", type=int, default=2000)
p.add_argument("--out", default=None, help="default: data/synthetic/<config>")
a = p.parse_args()
out = a.out or f"data/synthetic/{a.config}"
counts = generate(a.config, a.seed, a.n_train, a.n_dev, a.n_test, out)
print(f"wrote {out}: {counts}")
