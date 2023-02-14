#!/usr/bin/env bash

# Train multi-task models using the best validation hyperparameters.
set -e

log_base2="runs/train_best_hps_multi/"
n_runs=3  # shards multiplies the runs

echo "Logging in log_base=$log_base2"

for lr in 1e-5; do
  python -m maud.mega_tune -l $lr -r $n_runs -e 6 -i 6 -m roberta-base \
    --log_root $log_base2/roberta-base-lr-$lr --eval_split test
  python -m maud.mega_tune -l $lr -r $n_runs -e 6 -i 6 -m nlpaueb/legal-bert-base-uncased \
    --log_root $log_base2/legal-bert-lr-$lr --eval_split test
  python -m maud.mega_tune -l $lr -r $n_runs -e 6 -i 6 -m microsoft/deberta-v3-base \
    --log_root $log_base2/deberta-lr-$lr --eval_split test
done
