# Merger Agreement Understanding Dataset (MAUD)

This repository contains code for the [Merger Agreement Understanding Dataset (MAUD)](https://www.atticusprojectai.org/maud), a dataset for merger agreement review curated by the Atticus Project, and used im the 2021 American Bar Association Public Target Deal Points Study. It is part of the associated paper [MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding](https://arxiv.org/abs/TBD).


<img align="center" src="main_figure_60.png" width="1000">

## Installation
First, install pytorch with GPU support for your distribution: https://pytorch.org/get-started/locally/
Then, run `pip install -e .`

## Best Found Hyperparameters
Best found hyperparameters and corresponding validation scores, are available in the CSVs `best_found_hps/*.csv`.

## Training and Evaluation
Run `train.sh`, `train_multi.sh`, and `evaluate.sh` to train and evaluate models on best found hyperparameters.
