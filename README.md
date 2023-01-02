# Merger Agreement Understanding Dataset (MAUD)

## Installation
First, install pytorch with GPU support for your distribution: https://pytorch.org/get-started/locally/
Then, run `pip install -e .`

## Best Found Hyperparameters
Best found hyperparameters and corresponding validation scores, are available in the CSVs `best_found_hps/*.csv`.

## Training and Evaluation
Run `train.sh`, `train_multi.sh`, and `evaluate.sh` to train and evaluate models on best found hyperparameters.
