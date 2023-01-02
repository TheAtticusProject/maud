#!/usr/bin/env bash

python scripts/evaluate_plots.py --show_aupr runs/train_best_hps
python scripts/evaluate_plots.py --show_aupr runs/train_best_hps_multi
