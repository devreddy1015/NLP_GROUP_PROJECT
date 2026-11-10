"""M1 / M2: MuRIL or XLM-R fine-tuned for token classification. The first subword of each token carries its tag.

Trains once and scores every --eval file with that one model: a GPU run does not repeat bit for bit, so retraining
per eval file (as the CRF script does) would compare different models.

Run: uv run python -m src.models.transformer --model muril --train data/synthetic/full/train.jsonl --seed 42 \
         --eval data/synthetic/full/dev.jsonl data/real/dev.jsonl \
         --run-id YYYYMMDD-muril-synfull-dev-s42 YYYYMMDD-muril-synfull-realdev-s42
"""
import argparse
import time
from pathlib import Path

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

from src.data import TAGS, read_jsonl
from src.eval import evaluate, write_predictions, write_results

MODELS = {"muril": "google/muril-base-cased", "xlmr": "FacebookAI/xlm-roberta-base"}
# hub commits, pinned so a run can be repeated (checked 11 Nov 2026); any other name or a local folder loads as it is
REVISIONS = {"google/muril-base-cased": "afd9f36c7923d54e97903922ff1b260d091d202f",
             "FacebookAI/xlm-roberta-base": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"}
# fixed on 11 Nov 2026 with MuRIL on the full-noise config (synthetic dev, seed 42; logs.md): the cheapest setting within
# 0.002 micro F1 of the best. Every M1 / M2 run reuses them; no run tunes them or picks a checkpoint (initials.md §7)
EPOCHS, LR, BATCH_SIZE, MAX_LEN, WARMUP, WEIGHT_DECAY, CLIP = 3, 1e-4, 32, 256, 0.1, 0.01, 1.0
# mixed precision (bf16 autocast, so no gradient scaler) when training on a GPU: 1.7 times faster with the same dev F1.
# A CPU run stays in full precision
AMP = True
_TAG_ID = {tag: i for i, tag in enumerate(TAGS)}


def _first_subwords(word_ids: list[int | None]) -> list[tuple[int, int]]:
    """(position, token index) of each token's first subword; special tokens and later subwords are left out."""
    return [(pos, w) for pos, w in enumerate(word_ids) if w is not None and (pos == 0 or word_ids[pos - 1] != w)]


def encode(records: list[dict], tok, max_len: int = MAX_LEN) -> list[dict]:
    """Subword ids plus one label per subword: the tag on a token's first subword, -100 (skipped by the loss) elsewhere."""
    enc = tok([r["tokens"] for r in records], is_split_into_words=True, truncation=True, max_length=max_len)
    rows = []
    for i, r in enumerate(records):
        labels = [-100] * len(enc["input_ids"][i])
        for pos, w in _first_subwords(enc.word_ids(i)):
            labels[pos] = _TAG_ID[r["tags"][w]]
        rows.append({**{k: v[i] for k, v in enc.items()}, "labels": labels})
    return rows


def train(records: list[dict], checkpoint: str, seed: int = 42, epochs: int = EPOCHS, lr: float = LR,
          batch_size: int = BATCH_SIZE, max_len: int = MAX_LEN, device: str | None = None):
    """Fine-tune `checkpoint` on `records`. Returns (model, tokenizer, mean loss per epoch)."""
    transformers.set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    revision = REVISIONS.get(checkpoint)
    tok = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    net = AutoModelForTokenClassification.from_pretrained(checkpoint, revision=revision, num_labels=len(TAGS),
                                                          id2label=dict(enumerate(TAGS)), label2id=_TAG_ID).to(device)
    loader = DataLoader(encode(records, tok, max_len), batch_size=batch_size, shuffle=True,
                        collate_fn=DataCollatorForTokenClassification(tok),
                        generator=torch.Generator().manual_seed(seed))  # batch order depends on the seed alone, not on the model
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    steps = epochs * len(loader)
    sched = transformers.get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    kind = torch.device(device).type
    net.train()
    losses = []
    for _ in range(epochs):
        total = 0.0
        for batch in loader:
            with torch.autocast(kind, dtype=torch.bfloat16, enabled=AMP and kind == "cuda"):
                loss = net(**{k: v.to(device) for k, v in batch.items()}).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
        losses.append(total / len(loader))
    return net, tok, losses


@torch.no_grad()
def predict(net, tok, records: list[dict], batch_size: int = 64, max_len: int = MAX_LEN) -> list[list[str]]:
    net.eval()
    out = []
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        enc = tok([r["tokens"] for r in chunk], is_split_into_words=True, truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        best = net(**{k: v.to(net.device) for k, v in enc.items()}).logits.argmax(-1).tolist()
        for j, r in enumerate(chunk):
            tags = ["O"] * len(r["tokens"])  # a token past max_len, or one the tokenizer drops (zero-width space), stays O
            for pos, w in _first_subwords(enc.word_ids(j)):
                tags[w] = TAGS[best[j][pos]]
            out.append(tags)
    return out


def run(train_path, eval_paths, run_ids, model: str = "muril", seed: int = 42, results_dir="results",
        bootstrap_n: int = 1000, device: str | None = None, epochs: int = EPOCHS, lr: float = LR,
        batch_size: int = BATCH_SIZE, max_len: int = MAX_LEN, save_dir=None) -> list[dict]:
    # a wrong seed suffix or a repeated id would overwrite another run's results file
    if len(run_ids) != len(eval_paths) or len(set(run_ids)) != len(run_ids) or not all(i.endswith(f"-s{seed}") for i in run_ids):
        raise ValueError(f"need one distinct run id per eval file, each ending in -s{seed}; got {list(run_ids)}")
    tr, evals = read_jsonl(train_path), [read_jsonl(p) for p in eval_paths]  # before training: a mistyped path must not cost a GPU run
    if not tr or not all(evals):
        raise ValueError("the train file or an eval file is empty")
    train_ids = {r["id"] for r in tr}
    if any(r["id"] in train_ids for ev in evals for r in ev):  # the frozen test set must never be trained on (CLAUDE.md rule 2)
        raise ValueError("a training address is also in an eval file")
    checkpoint = MODELS.get(model, model)
    start = time.perf_counter()
    net, tok, losses = train(tr, checkpoint, seed, epochs, lr, batch_size, max_len, device)
    if save_dir:  # about 1 GB for MuRIL: only for a model that will be used again, such as the demo's
        net.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
    config = {"train": str(train_path), "n_train": len(tr), "checkpoint": checkpoint,
              "revision": REVISIONS.get(checkpoint), "epochs": epochs, "lr": lr, "batch_size": batch_size,
              "max_len": max_len, "warmup": WARMUP, "weight_decay": WEIGHT_DECAY, "clip": CLIP,
              "amp": AMP and net.device.type == "cuda", "epoch_loss": losses,
              "train_seconds": round(time.perf_counter() - start),
              "device": torch.cuda.get_device_name() if net.device.type == "cuda" else "cpu",
              "torch": torch.__version__, "transformers": transformers.__version__}
    results = []
    for eval_path, run_id, ev in zip(eval_paths, run_ids, evals):
        pred = predict(net, tok, ev, max_len=max_len)
        data = f"{Path(eval_path).parent.name}/{Path(eval_path).stem}"
        res = evaluate(ev, pred, run_id=run_id, model=Path(model).name, data=data, seed=seed, bootstrap_n=bootstrap_n,
                       config=config)
        write_results(res, results_dir)
        write_predictions(ev, pred, run_id, results_dir)  # this model cannot be rebuilt exactly, so its tags are kept
        results.append(res)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="muril", help="muril, xlmr, or any Hugging Face name or local folder")
    p.add_argument("--train", required=True)
    p.add_argument("--eval", required=True, nargs="+")
    p.add_argument("--run-id", required=True, nargs="+", help="one per --eval file, in the same order")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--save-dir", default=None, help="keep the trained model here (about 1 GB), e.g. models/muril for the demo")
    a = p.parse_args()
    for r in run(a.train, a.eval, a.run_id, a.model, a.seed, a.results_dir, save_dir=a.save_dir):
        print(f"{r['run_id']}: n={r['n']} span micro F1={r['span']['micro']['f1']:.3f} macro F1={r['span']['macro']['f1']:.3f} "
              f"exact={r['exact_match']:.3f} CI95={r['ci95'][0]:.3f}-{r['ci95'][1]:.3f} train={r['config']['train_seconds']}s")
