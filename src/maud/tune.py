import argparse
import contextlib
import math
import pathlib
import pickle
from typing import Dict, List, Optional

from kornia.losses import focal
import pandas as pd
import numpy as np
import sklearn.metrics
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.nn import functional as F
import tqdm
import transformers

import maud.specs
from maud import auto_load_data, category_utils, data, data_utils, specs, utils


def R(unrounded_list, digits=3) -> list:
    return [round(x, digits) for x in unrounded_list]


def log(text: str = "", newline="\n", stdout=True, file_write=False, *, path="runs.txt"):
    # Stopping file write for now because parallelism makes this file messy.
    if file_write:
        with open(path, "a") as f:
            f.write(text + newline)
    if stdout:
        print(text)


def main(
        spec: maud.specs.BaseDatasetSpec,
        args: argparse.Namespace,
) -> None:
    # Either we have epochs. Or we have eval_interval
    use_fixed_batches_wrapper = args.num_updates is not None
    if use_fixed_batches_wrapper:
        # Update with a fixed number of batches
        # Since each question dataset can have different number of
        # updates per epoch, especially if we use oversampling.
        assert ":" in args.num_updates
        num_updates, batches_per_iter = args.num_updates.split(":")
        num_updates = int(num_updates)
        batches_per_iter = int(batches_per_iter)
        assert num_updates > 0
        assert batches_per_iter > 0
        n_epochs = math.ceil(num_updates / batches_per_iter)
        n_runs = args.nruns
    else:
        # Train for a certain number of epochs (standard)
        n_epochs = args.nepochs
        n_runs = args.nruns

    eval_interval = args.eval_interval
    if args.dry_run:
        # Override some settings for code testing mode..
        n_epochs = 1
        n_runs = 1
        eval_interval = 1

    log()
    if args.dry_run:
        log("**DRY RUN**\n")
    log(f'DATASET={spec.id}/"{spec.answer_key}", valid_prop={args.valid_prop}')
    log(repr(spec))
    log()
    log('full args: {}\n'.format(args))

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    if 'deberta-v3' in tokenizer.name_or_path:
        # This wasn't configured by the maintainers.
        tokenizer.max_length = 512
        tokenizer.model_max_length = 512
    if args.auto_max_seq_len:
        auto_len = category_utils.get_spec_id_to_max_tokens(
            spec.id, tokenizer) + 1
        print(f"Automatically setting the max sequence len to {auto_len}")
        assert tokenizer.model_max_length >= auto_len
        auto_len = category_utils.get_spec_id_to_max_tokens(spec.id, tokenizer) + 1
        tokenizer.model_max_length = auto_len
    elif args.max_seq_len is not None:
        assert args.max_seq_len >= 2
        tokenizer.model_max_length = args.max_seq_len
    print(f"Max sequence length: {tokenizer.model_max_length}")

    for run in range(1, n_runs + 1):
        run_dir = args.log_root / f"q{spec.id}_{args.model}"
        run_dir.mkdir(exist_ok=True, parents=True)

        if args.eval_split == "valid":
            train_df = data_utils.load_df("train")
            test_df = data_utils.load_df("dev")
        elif args.eval_split == "test":
            train_df = data_utils.load_df("train")
            test_df = data_utils.load_df("test")
        else:
            raise ValueError(args.eval_split)

        if args.drop_additional:
            print("[SPECIAL] Drop additional.")
            mask = train_df["data_type"] != "abridged"
            assert np.sum(mask) > 0
            train_df = train_df[mask]
            if args.eval_split != "test":
                mask = test_df["data_type"] != "abridged"
                assert np.sum(mask) > 0
                test_df = test_df[mask]

        train_ds = data_utils.df_to_records(
            train_df,
            question=spec.answer_key,
            subquestion=spec.sub_question_key,
        )
        test_ds = data_utils.df_to_records(
            test_df,
            question=spec.answer_key,
            subquestion=spec.sub_question_key,
        )
        print(f"{len(train_ds)} train records loaded from disk ({spec.n_classes} classes)")
        print(f"{len(test_ds)} test records loaded from disk ({spec.n_classes} classes)")

        print(f"{len(train_ds)} records will be used for the training set")
        print(f"{len(test_ds)} records will be used for the eval set")

        # Used for saved pickle later. Make sure that we don't shuffle test_ds, otherwise the order here
        #   will be wrong.
        test_ds_data_types = [rec['data_type'] for rec in test_ds]

        # Drop data for "function as dataset" experiments.
        if args.partial_data is not None:
            new_train_ds = data.keep_data_part(train_ds, keep_prop=args.partial_data)
            print(f"[ABLATION] Keeping only {args.partial_data*100:.2f}% of the dataset.")
            print("[ABLATION] Dropped {} samples from training, now have {} samples remaining.".format(
                len(train_ds) - len(new_train_ds), len(new_train_ds)))
            train_ds = new_train_ds

        # Tokenize
        train_ds = specs._encode_records_as_dataset(train_ds, tokenizer, dry_run=args.dry_run)
        test_ds = specs._encode_records_as_dataset(test_ds, tokenizer, dry_run=args.dry_run)

        print(f"{len(train_ds)} examples put in train_ds")
        print(f"{len(test_ds)} examples put in test_ds")
        train_ds_counts = data.count_labels(train_ds, spec.ordered_answers)
        test_ds_counts = data.count_labels(test_ds, spec.ordered_answers)

        if len([count for _, count in train_ds_counts if count == 0]) > 0:
            raise ValueError(f"BAD JOB: Not enough unique labels in train {train_ds_counts}")

        if len([count for _, count in test_ds_counts if count == 0]) > 0:
            raise ValueError(f"BAD JOB: Not enough unique labels in test {test_ds_counts}")

        log(f"train: {train_ds_counts}", file_write=run==1)
        log(f"test: {test_ds_counts}", file_write=run==1)

        if args.oversample:
            log(f"OVERSAMPLING...", file_write=run==1)
            train_ds = data.oversample(train_ds)
            print(f"{len(train_ds)} examples in oversampled train_ds")
            log(f"oversampled train: {data.count_labels(train_ds, spec.ordered_answers)}", file_write=run==1)

        if args.focal_gamma is not None:
            log(f"FOCAL LOSS: gamma={args.focal_gamma}")

        if args.oversample is None and args.focal_gamma is None:
            log(f"NO ALGORITHMIC DATA IMBALANCE CORRECTIONS.", file_write=run==1)

        if args.dry_run:
            return

        train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        if use_fixed_batches_wrapper:
            train_dataloader = utils.DataLoaderFixedBatches(
                train_dataloader,
                batches_per_iter=batches_per_iter,
            )
            assert len(train_dataloader) == batches_per_iter

        model = utils.load_model(args, num_labels=spec.n_classes)
        optimizer = utils.make_optimizer(model, args=args)

        best_test_accuracy = -1

        print(spec)
        log(f"RUN {run} OUT OF {n_runs}")

        assert n_epochs >= 1
        for epoch in range(1, n_epochs + 1):
            print()
            train(model, optimizer, train_dataloader,
                  epoch=epoch,
                  num_classes=spec.n_classes,
                  half_precision=args.amp,
                  focal_loss_gamma=args.focal_gamma)
            batch_num = len(train_dataloader) * epoch  # Each batch is one update to the model.
            if use_fixed_batches_wrapper:
                log(f"Batch number {batch_num}::")
            else:
                log(f"Epoch {epoch}")
            if epoch % eval_interval == 0:
                train_acc, _, _, train_f1_all, *_ = evaluate(model, train_dataloader, name="train")
                test_acc, f1_micro, f1_macro, test_f1_all, precisions, recalls, logits, labels = evaluate(
                    model, test_dataloader, name="test")
                best_test_accuracy = max(test_acc, best_test_accuracy)

                log(f"Answers (in same order as F1-scores): {spec.ordered_answers}")
                log(f"final train F1-scores (unaveraged): {R(train_f1_all)}")
                log(f"final test F1-scores (unaveraged): {R(test_f1_all)}")
                log(f"final test precisions (unaveraged): {R(precisions)}")
                log(f"final test recalls (unaveraged): {R(recalls)}")
                log(f"final test accuracy: {test_acc:.3f}")
                log(f"best test accuracy: {best_test_accuracy:.3f}")
                log(f"final test F1-score (micro/macro): {f1_micro:.3f}/{f1_macro:.3f}")

                run_results = dict(
                    version=1,
                    epoch=epoch,
                    batch_num=batch_num,
                    run_num=run,
                    n_runs=n_runs,
                    spec_id=spec.id,
                    q=spec,
                    spec=spec,
                    args=args,
                    labels=labels,
                    logits=logits,
                    train_ds_counts=train_ds_counts,
                    test_ds_counts=test_ds_counts,
                    train_acc=train_acc,
                    train_f1_all=train_f1_all,
                    test_f1_all=test_f1_all,
                    precisions=precisions,
                    recalls=recalls,
                    test_acc=test_acc,
                    best_test_accuracy=best_test_accuracy,
                    f1_micro=f1_micro,
                    f1_macro=f1_macro,
                    test_data_types=test_ds_data_types,
                )

                # Write run_results
                timestamp = utils.make_unique_filename()
                run_save_path = run_dir / f"{timestamp}_run_results.pkl"
                with open(run_save_path, "wb") as f:
                    pickle.dump(run_results, f)
                log(f"Dumped run_results to '{run_save_path}'")

                if args.save:
                    run_dir.mkdir(exist_ok=True, parents=True)
                    # Save model checkpoint (could be large!)
                    model_path = run_dir / "model.pkl"
                    torch.save(model.module.state_dict(), model_path)
                    log(f"Saved model to {model_path}")


def train(
        model,
        optimizer,
        train_dataloader,
        epoch,
        *,
        half_precision: bool,
        num_classes: int,
        focal_loss_gamma: Optional[float] = None,
        log_interval=10,
) -> None:
    model.train()
    if focal_loss_gamma is None:
        criterion = torch.nn.BCEWithLogitsLoss()
        convert_one_hot = True
    else:
        criterion = focal.FocalLoss(alpha=0.5, gamma=focal_loss_gamma, reduction="mean")
        convert_one_hot = False

    scaler = amp.GradScaler()
    if half_precision:
        print("Using half precision!")

    # Loop over each batch from the training set
    for step, batch in enumerate(train_dataloader):
        model_inputs = {k: v.cuda() for k, v in batch.items()
                        if k in data.MODEL_INPUT_KEYS}
        labels = batch["label"].cuda()

        if convert_one_hot:
            target = F.one_hot(labels, num_classes=num_classes).cuda().float()
        else:
            target = labels

        # Zero gradient buffers
        optimizer.zero_grad()

        # Forward pass
        if half_precision:
            autocast = amp.autocast(dtype=torch.float16)
        else:
            autocast = contextlib.nullcontext()

        with autocast:
            output = model(**model_inputs)[0].squeeze()
            if output.shape[0]==target.shape[1]:
                output = output.reshape(target.shape)
            loss = criterion(output, target)

        # Backward pass and Update weights
        if half_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if step % log_interval == 0 and step > 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(batch["label"]),
                len(train_dataloader.dataset),
                   100. * step / len(train_dataloader), loss))


def evaluate(model, test_hard_dataloader, name=None):
    model.eval()
    all_predictions = []
    all_labels = []
    all_logits = []

    for batch in test_hard_dataloader:
        model_inputs = {k: v.cuda() for k, v in batch.items()
                        if k in data.MODEL_INPUT_KEYS}

        with torch.no_grad():
            logits = model(**model_inputs)[0]

        predictions_b: np.ndarray = torch.argmax(logits, dim=1).detach().cpu().numpy()
        # predictions = (output > 0).astype(int)  # sigmoid version

        labels_b: np.ndarray = batch["label"].detach().cpu().numpy()
        all_predictions.append(predictions_b)
        all_labels.append(labels_b)
        all_logits.append(logits.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    all_logits = np.concatenate(all_logits)
    acc = np.mean(all_labels == all_predictions)
    # f1_labels = list(set(all_labels))
    # f1_micro = sklearn.metrics.f1_score(all_labels, all_predictions, labels=f1_labels, average='micro')
    # f1_macro = sklearn.metrics.f1_score(all_labels, all_predictions, labels=f1_labels, average='macro')
    # f1_all = sklearn.metrics.f1_score(all_labels, all_predictions, labels=f1_labels, average=None)
    # precision = sklearn.metrics.precision_score(all_labels, all_predictions, labels=f1_labels, average=None)
    # recall = sklearn.metrics.recall_score(all_labels, all_predictions, labels=f1_labels, average=None)
    # Calculating AUPR here? Nah.
    f1_micro = sklearn.metrics.f1_score(all_labels, all_predictions, average='micro')
    f1_macro = sklearn.metrics.f1_score(all_labels, all_predictions, average='macro')
    f1_all = sklearn.metrics.f1_score(all_labels, all_predictions, average=None)
    precision = sklearn.metrics.precision_score(all_labels, all_predictions, average=None)
    recall = sklearn.metrics.recall_score(all_labels, all_predictions, average=None)

    if name is not None:
        print(f"[{name}] ", end='')
    print(f'Accuracy: {acc:.4f}  F1: {R(f1_all, 4)}  F1_macro: {f1_macro:.4f}')
    return acc, f1_micro, f1_macro, f1_all, precision, recall, all_logits, all_labels


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="roberta-base")
    parser.add_argument("--log_root", type=pathlib.Path, default="runs/default")
    parser.add_argument("--valid_prop", "-p", type=float, default=0.2)
    parser.add_argument("--nruns", "-r", type=int, default=1)
    parser.add_argument("--eval_split", default="valid", choices=["valid", "test"])

    parser.add_argument("--ngpus", "-n", type=int, default=1)
    parser.add_argument("--dry_run", "-d", action="store_true")

    parser.add_argument("--partial_data", type=float, default=None)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp autocasting to reduce compute/memory.")

    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--weight_decay", "-w", type=float, default=0.01)
    parser.add_argument("--learning_rate", "-l", type=float, default=2e-5)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--grid_search", "-g", action="store_true")
    parser.add_argument("--save", "-S", action="store_true", help="Save checkpoints. (warning: usually large!)")
    parser.add_argument("--oversample", "-o", action="store_true")
    parser.add_argument("--focal_gamma", "-f", type=float, default=None)

    parser.add_argument("--use_add", "-a", action="store_true", help="Add abridged data augmentation")
    parser.add_argument("--use_synth", "-s", action="store_true", help="Add rare answers augmentation.")
    # Num updates flag.
    # But what about evaluation period?
    # Then we need to set another flag. What about we do colon notation.
    epoch_or_updates_group = parser.add_mutually_exclusive_group()
    epoch_or_updates_group.add_argument("--nepochs", "-e", type=int, default=2)
    epoch_or_updates_group.add_argument("--num_updates", type=str,
                                        help="A string of the form NUM_UPDATES/UPDATES_PER_EVAL. e.g. 1000:200"
                                        )
    parser.add_argument("--eval_interval", "-i", type=int, default=1)

    seq_group = parser.add_mutually_exclusive_group()
    seq_group.add_argument("--auto_max_seq_len", action="store_true")
    seq_group.add_argument("--max_seq_len", type=int, default=None)

    parser.add_argument("--shard", type=str, help="String of the form SHARD_IDX/NUM_SHARDS e.g. 0/10 or 9/10")

    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("-q", "--append_question_num", action="append", type=int,
                                 help="Add a question num to train on")
    selection_group.add_argument("--spec_id", action="append", type=str, help="Select question/spec by ID")
    selection_group.add_argument("--all_questions", "-A", action="store_true")
    selection_group.add_argument("-B", "--bigbird_by_group", type=str,
                                 choices=["small", "mid", "balrog", "saruman"])

    parser.add_argument("-T", "--load_test_best_hps_path", type=pathlib.Path, default=None)
    parser.add_argument("--drop_additional", action="store_true")
    return parser


def build_spec_id_to_best_tune_kwargs(
        df_best_hps,
        model_name: str,
        partial: Optional[float],
        kept_kwargs: Optional[dict] = None,
) -> Dict[str, dict]:

    # Use output to filter by requested spec_id.
    mask = df_best_hps["model_name"] == model_name
    if partial is not None:
        assert "partial" in df_best_hps.columns
        partial_mask = df_best_hps["partial"] == partial
        mask = mask & partial_mask

    our_df_best = df_best_hps[mask]
    assert len(our_df_best) > 0
    available_spec_ids = set(our_df_best['spec_id'])
    # Check that each row has a unique spec_id.
    assert len(available_spec_ids) == len(our_df_best)

    n_rows = len(our_df_best)
    spec_id_to_kwargs = {}
    for i in tqdm.tqdm(range(n_rows)):
        row = our_df_best.iloc[i]
        model_name = row["model_name"]
        spec_id = row["spec_id"]
        spec_id = spec_id.replace(".", "-")  # For compatibility
        num_updates = row["update_num"]
        learning_rate = row["lr"]
        args_text = (f"-m {model_name} --spec_id {spec_id} "
                     f"-l {learning_rate} --num_updates {num_updates}:{num_updates} "
                     f"--eval_split test "
                     f"-oas "
                     )
        parser = get_parser()
        args = parser.parse_args(args_text.split())
        for k, v in kept_kwargs.items():
            setattr(args, k, v)
        spec = category_utils.get_valid_spec_with_id(spec_id)
        kwargs = dict(args=args, spec=spec)
        spec_id_to_kwargs[spec.id] = kwargs
    return spec_id_to_kwargs


def console_main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    our_specs: List[maud.specs.BaseDatasetSpec] = []
    if args.all_questions:
        our_specs = auto_load_data.get_all_valid_specs(verbosity=1)
    elif args.bigbird_by_group:
        assert "google/bigbird-roberta-base" == args.model, args.model
        spec_ids_by_gpu_name = category_utils.get_bigbird_spec_ids_by_gpu_name()
        our_spec_ids = spec_ids_by_gpu_name[args.bigbird_by_group]
        assert len(set(our_spec_ids)) == len(our_spec_ids)
        all_specs = auto_load_data.generate_specs(verbosity=1)
        for spec in all_specs:
            if spec.id in our_spec_ids:
                our_specs.append(spec)
        assert len(our_spec_ids) == len(our_specs)
    elif args.spec_id:
        our_spec_ids = set()
        for s in args.spec_id:
            for _spec_id in s.split(","):
                our_spec_ids.add(_spec_id)
        all_specs = auto_load_data.generate_specs(verbosity=1)
        for spec in all_specs:
            if spec.id in our_spec_ids:
                our_specs.append(spec)
    else:
        all_specs = auto_load_data.generate_specs(verbosity=1)
        q_indices = args.append_question_num or [0]
        for q_num in q_indices:
            our_specs.append(all_specs[q_num])

    print(f"Selected {len(our_specs)} question dataset(s) to train on.")

    if args.shard:
        shard_idx, n_shards = [int(x) for x in args.shard.split("/")]
        assert n_shards >= 1
        assert shard_idx in range(n_shards)
        print(f"Sharding mode activated. shard_idx={shard_idx} out of n_shards={n_shards}")
        our_specs = our_specs[shard_idx::n_shards]
        print(f"Kept {len(our_specs)} question dataset(s) for this shard..")

    REPORT_ERRORS = False

    if args.load_test_best_hps_path is not None:
        print("[SPECIAL] Test mode. Overwriting all hyperparams with best hyperparam settings.")
        print(f"[SPECIAL] model_name={args.model}")
        print(f"[SPECIAL] Attempting to load spec_ids={[spec.id for spec in our_specs]}")
        df_best_hps_path: pathlib.Path = args.load_test_best_hps_path
        assert df_best_hps_path.is_file()
        df_best_hps = pd.read_csv(df_best_hps_path, dtype={"spec_id": str})

        # Keep: auto_max_seq_len, batch_size, n_runs, o, a, s
        kept_keys = ["auto_max_seq_len", "batch_size", "nruns",
                     "oversample", "use_add", "use_synth", "log_root",
                     "partial_data", "model",
                     ]
        kept_kwargs = {k: getattr(args, k) for k in kept_keys}

        spec_id_to_best_kwargs = build_spec_id_to_best_tune_kwargs(
            df_best_hps,
            model_name=args.model,
            partial=args.partial_data,
            kept_kwargs=kept_kwargs,
        )
        best_kwargs: List[dict] = []
        for spec in our_specs:
            best_kwargs.append(spec_id_to_best_kwargs[spec.id])

        for kwargs in tqdm.tqdm(best_kwargs, desc="multiple question specs (best kwargs))"):
            main(**kwargs)
    else:
        for spec in tqdm.tqdm(our_specs, desc="multiple question specs"):
            try:
                main(spec, args)
            except ValueError as e:
                if REPORT_ERRORS:
                    log(f"FAIL: (ValueError {e}) \n\t{spec} \n\t{spec.answer_counter_all()}", file_write=True,
                        # path="errors_lenient.txt")
                        path="errors.txt")
                else:
                    raise


if __name__ == "__main__":
    console_main()