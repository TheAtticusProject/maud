## Best Hyperparameters and Validation Scores


### Multi-task Validation results
* `multi.txt` and `multi_task_no_abridged.md` contain validation scores for multi-task training.

### Single-task Validation results
* All directories `basic_*` contain best hyperparameters at `df_best_hps_spec_id.csv`, and a full list of mean validation scores in `mean_scores_by_hp.csv`.
  * `basic_legal_berts` contains LegalBERT validation results.
  * `basic_bb_post_contract_800_v0` contains BigBird results.
  * `basic_partial_post_contract_v1` contains RoBERTa dataset-size ablation results.
  * `basic_post_contract_v1` contains all other single-task results.
