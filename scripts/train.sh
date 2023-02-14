#!/usr/bin/env bash
# Script for training single-task models given the best validation hyperparameters.

set -e
log_base="runs/train_best_hps"
n_runs=3
echo "Logging in log_base=$log_base"

script_base="python src/maud/tune.py -A -r $n_runs -oas"
script="$script_base --load_test_best_hps_path best_found_hps/basic_post_contract_v1/df_best_hps_spec_id.csv"
script_legal="$script_base --load_test_best_hps_path best_found_hps/basic_legal_berts_v1/df_best_hps_spec_id.csv"
script_bb="$script_base --load_test_best_hps_path best_found_hps/basic_bb_post_contract_800_v0/df_best_hps_spec_id.csv"


# Train BERT
$script -m bert-base-cased --log_root $log_base/bert-base-cased

# Train RoBERTa
# $script -m roberta-base --log_root $log_base/roberta-base

# Train DeBERTa
# $script -m microsoft/deberta-v3-base --log_root $log_base/deberta-v3-base

# Train BERT
# $script_legal -m nlpaueb/legal-bert-base-uncased --log_root $log_base/legal-bert-base-uncased

# Train BigBird
# $script_bb -m google/bigbird-roberta-base --log_root $log_base/bigbird
