#!/usr/bin/env bash
# E2, E3 and E4: one training run per config and seed (build-plan step 12). No framework, just loops.
#
#   scripts/run_experiments.sh --dry-run          # list the 48 runs without starting any
#   scripts/run_experiments.sh e2 e3              # some experiments: e2 | e3 | e4 | xlmr | all (default)
#   scripts/run_experiments.sh --final all        # also score the real TEST set: final runs only (CLAUDE.md rule 2)
#
# Each run takes 5 to 6 minutes on the RTX 4060 (an estimate: 314 s of training plus loading and scoring), so all 48
# take four to five hours: ask the team first (rule 7).
# A run whose last results file exists is skipped, so the script can be stopped and started again on the same day;
# on a later day pass the first day's date: DATE=20261120 scripts/run_experiments.sh e3
# A --final run trains again (models are not kept) and so writes new dev results beside the test results. The tables
# use the newest run of each configuration and seed, so nothing is counted twice; update the run table in logs.md.
# Not included: the clean + one-family runs (only-N1 .. only-N8), still an open decision (build-plan C3).
set -euo pipefail
cd "$(dirname "$0")/.."

DATE=${DATE:-$(date +%Y%m%d)}
SEEDS=(42 43 44)   # training seeds
DATA_SEED=42       # the synthetic data is generated once and shared by the three training seeds
KS=(25 50 100 200) # real addresses added in E4; k = 0 is E2's full-noise run
RESULTS=${RESULTS:-results}

DRY=0 FINAL=0 WHAT=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --final) FINAL=1 ;;
    e2 | e3 | e4 | xlmr) WHAT+=("$arg") ;;
    all) WHAT=(e2 e3 e4 xlmr) ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
if [ "${#WHAT[@]}" = 0 ]; then WHAT=(e2 e3 e4 xlmr); fi

say() { if [ "$DRY" = 1 ]; then echo "$*"; else echo "+ $*" >&2; "$@"; fi; }

need() { [ "$DRY" = 1 ] || [ -f "$1" ] || { echo "missing $1: $2" >&2; exit 1; }; }
need data/real/dev.jsonl "annotate and export the real dev set first (src.annotate export)"
if [ "$FINAL" = 1 ]; then need data/real/test.jsonl "the real test set is not exported yet"; fi
if [[ " ${WHAT[*]} " == *" e4 "* ]] && [ "$DRY" = 0 ]; then
  need data/real/train.jsonl "E4 draws its k addresses from the real-train pool"
  pool=$(wc -l < data/real/train.jsonl)  # checked now, not after hours of E2 and E3
  most=${KS[${#KS[@]}-1]}  # not ${KS[-1]}: macOS ships bash 3.2
  [ "$pool" -ge "$most" ] || { echo "data/real/train.jsonl holds $pool addresses; E4 needs $most" >&2; exit 1; }
fi

synthetic() { # config [n_train]: generate the set unless it is already there
  local config=$1 n=${2:-20000} dir=data/synthetic/$1
  if [ "$n" != 20000 ]; then dir=data/synthetic/$1-40k; fi
  [ -f "$dir/train.jsonl" ] || say uv run python -m src.generate --config "$config" --seed "$DATA_SEED" --n-train "$n" --out "$dir"
}

train() { # model tag train_file synthetic_config seed
  local model=$1 tag=$2 train_file=$3 config=$4 seed=$5
  local evals=("data/synthetic/$config/test.jsonl" data/real/dev.jsonl)
  local ids=("$DATE-$model-$tag-syntest-s$seed" "$DATE-$model-$tag-realdev-s$seed")
  if [ "$FINAL" = 1 ]; then evals+=(data/real/test.jsonl) ids+=("$DATE-$model-$tag-realtest-s$seed"); fi
  if [ -f "$RESULTS/${ids[${#ids[@]}-1]}.json" ]; then echo "skip $model $tag s$seed: results exist" >&2; return; fi
  say uv run python -m src.models.transformer --model "$model" --train "$train_file" --seed "$seed" \
    --eval "${evals[@]}" --run-id "${ids[@]}" --results-dir "$RESULTS"
}

e2() { # clean against full noise
  for config in clean full; do
    synthetic "$config"
    for seed in "${SEEDS[@]}"; do train muril "syn$config" "data/synthetic/$config/train.jsonl" "$config" "$seed"; done
  done
}

e3() { # full noise minus one family at a time; the reference is E2's full run
  for n in 1 2 3 4 5 6 7 8; do
    synthetic "full-minus-N$n"
    for seed in "${SEEDS[@]}"; do
      train muril "synfullminusN$n" "data/synthetic/full-minus-N$n/train.jsonl" "full-minus-N$n" "$seed"
    done
  done
}

e4() { # full noise plus k real addresses, and 40k synthetic with none (the other side of H3)
  synthetic full
  synthetic full 40000
  for seed in "${SEEDS[@]}"; do
    for k in "${KS[@]}"; do
      local out=data/synthetic/mixed/full-k$k-s$seed/train.jsonl
      [ -f "$out" ] || say uv run python -m src.mix --synthetic data/synthetic/full/train.jsonl \
        --real data/real/train.jsonl --k "$k" --seed "$seed" --out "$out"
      train muril "synfullk$k" "$out" full "$seed"
    done
    train muril synfull40k data/synthetic/full-40k/train.jsonl full "$seed"
  done
}

xlmr() { # the second backbone on full noise
  synthetic full
  for seed in "${SEEDS[@]}"; do train xlmr synfull data/synthetic/full/train.jsonl full "$seed"; done
}

for what in "${WHAT[@]}"; do "$what"; done
