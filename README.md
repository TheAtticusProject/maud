# Merger Agreement Understanding Dataset (MAUD)

<img align="center" src="main_figure.png" width="1000">

This repository contains code for the [Merger Agreement Understanding Dataset (MAUD)](https://www.atticusprojectai.org/maud), a dataset for merger agreement review curated by the Atticus Project, and used in the 2021 American Bar Association Public Target Deal Points Study. It is part of the associated paper [MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding](https://arxiv.org/abs/TBD).


## Installation
First, install pytorch with GPU support for your distribution: https://pytorch.org/get-started/locally/

Then, run `pip install -e .`

Unzip the data files with `unzip data.zip`.

## Best Found Hyperparameters
Best found hyperparameters and corresponding validation scores, are available in the CSVs `best_found_hps/*.csv`.

## Training and Evaluation
Run `train.sh`, `train_multi.sh`, and `evaluate.sh` to train and evaluate models on best found hyperparameters.
