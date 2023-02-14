# Merger Agreement Understanding Dataset (MAUD)

<img align="center" src="main_figure.png" width="1000">

This repository contains code for the Merger Agreement Understanding Dataset (MAUD), a dataset for merger agreement review used in the 2021 American Bar Association Public Tagget Deal Points Study.


## Installation
First, install pytorch with GPU support for your distribution: https://pytorch.org/get-started/locally/

Then, run `pip install -e .`

Unzip the data files with `unzip data.zip`.

## Best Found Hyperparameters
Best found hyperparameters and corresponding validation scores, are available in the CSVs `best_found_hps/*.csv`.

## Training and Evaluation
Run `train.sh`, `train_multi.sh`, and `evaluate.sh` to train and evaluate models on best found hyperparameters.
