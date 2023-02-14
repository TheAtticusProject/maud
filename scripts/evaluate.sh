#!/usr/bin/env bash

# Evaluate single-task models.
python scripts/evaluate_plots.py --show_aupr runs/train_best_hps

# Evaluate multi-task models.
python scripts/evaluate_plots.py --show_aupr runs/train_best_hps_multi
