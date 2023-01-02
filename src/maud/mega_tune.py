import argparse
import contextlib
import math
import pathlib
import pickle
from typing import Dict, List, Optional, Tuple

import sklearn.metrics
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.nn import functional as F
import tqdm
import numpy as np
import transformers

import maud.specs
from maud import auto_load_data, category_utils, data, data_utils, utils
from maud.multi_head_model import MultitaskDataLoader


def R(unrounded_list, digits=3) -> list:
    return [round(x, digits) for x in unrounded_list]


def log(text: str = "", newline="\n", stdout=True, file_write=False, *, path="runs.txt"):
    # Stopping file write for now because parallelism makes this file messy.
    if file_write:
        with open(path, "a") as f:
            f.write(text + newline)
    if stdout:
        print(text)


def subroutine(train_df, test_df, specs, *, tokenizer, batch_size, dry_run: bool = False):
    spec_id_to_train_dl: Dict[str, DataLoader] = {}
    spec_id_to_test_dl: Dict[str, DataLoader] = {}
    spec_id_to_specs: Dict[str, maud.specs.BaseDatasetSpec] = {}
    for spec in specs:
        assert spec.id not in spec_id_to_specs.keys()
        spec_id_to_specs[spec.id] = spec
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
        train_ds = maud.specs._encode_records_as_dataset(train_ds, tokenizer, dry_run=dry_run)
        test_ds = maud.specs._encode_records_as_dataset(test_ds, tokenizer, dry_run=dry_run)
        print(f"spec '{spec.id}': {len(train_ds)} train records loaded from disk ({spec.n_classes} classes)")
        print(f"spec '{spec.id}': {len(test_ds)} test records loaded from disk ({spec.n_classes} classes)")
        train_ds_counts = data.count_labels(train_ds, spec.ordered_answers)
        test_ds_counts = data.count_labels(test_ds, spec.ordered_answers)

        if len([count for _, count in train_ds_counts if count == 0]) > 0:
            raise ValueError(f"BAD JOB: Not enough unique labels in train {train_ds_counts}")

        if len([count for _, count in test_ds_counts if count == 0]) > 0:
            raise ValueError(f"BAD JOB: Not enough unique labels in test {test_ds_counts}")
        print(f"train: {train_ds_counts}")
        print(f"test: {test_ds_counts}")
        print(f"OVERSAMPLING...")
        train_ds = data.oversample(train_ds)
        print(f"{len(train_ds)} examples in oversampled train_ds")
        log(f"oversampled train: {data.count_labels(train_ds, spec.ordered_answers)}")

        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        spec_id_to_train_dl[spec.id] = train_dataloader
        spec_id_to_test_dl[spec.id] = test_dataloader
    return spec_id_to_specs, spec_id_to_train_dl, spec_id_to_test_dl


def main(
        specs: List[maud.specs.BaseDatasetSpec],
        args: argparse.Namespace,
) -> None:
    # Either we have epochs. Or we have eval_interval
    n_epochs = args.nepochs
    n_runs = args.nruns
    eval_interval = args.eval_interval

    # SOMEDAY: Allow these guys to be arguments?
    n_dl_batches_per_epoch = 100
    n_specs_per_batch_list = 4

    if args.dry_run:
        # Override some settings for code testing mode..
        n_epochs = 1
        n_runs = 1
        eval_interval = 1

    log()
    if args.dry_run:
        log("**DRY RUN**\n")
    print(f"Multihead training with {len(specs)} specs")
    log()
    log('full args: {}\n'.format(args))

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    if 'deberta-v3' in tokenizer.name_or_path:
        # This wasn't configured by the maintainers. Adding in our best guess.
        tokenizer.max_length = 512
        tokenizer.model_max_length = 512
    print(f"Max sequence length: {tokenizer.model_max_length}")

    train_df = data_utils.load_df("train")
    if args.eval_split == "valid":
        test_df = data_utils.load_df("dev")
    elif args.eval_split == "test":
        test_df = data_utils.load_df("test")
    else:
        raise ValueError(args.eval_split)

    if args.drop_additional:
        print("[SPECIAL] Drop additional.")
        mask = train_df["data_type"] != "abridged"
        assert np.sum(mask) > 0
        print("[SPECIAL] Dropped {np.sum(~mask)} additional samples.")
        train_df = train_df[mask]
        if args.eval_split != "test":
            mask = test_df["data_type"] != "abridged"
            assert np.sum(mask) > 0
            test_df = test_df[mask]


    # Build train and test DataLoaders for each spec.
    spec_id_to_specs, spec_id_to_train_dl, spec_id_to_test_dl = subroutine(
        train_df, test_df, specs, tokenizer=tokenizer, batch_size=args.batch_size, dry_run=args.dry_run)

    for run in range(1, n_runs + 1):
        mt_train_dl = MultitaskDataLoader(spec_id_to_train_dl, n_dl_batches_per_epoch, n_specs_per_batch_list)

        run_dir = args.log_root / "multihead"
        run_dir.mkdir(exist_ok=True, parents=True)
        if args.dry_run:
            return

        model = utils.load_multi_head_model(args, specs)
        optimizer = utils.make_optimizer(model, args=args)
        log(f"RUN {run} OUT OF {n_runs}")

        assert n_epochs >= 1
        for epoch in tqdm.tqdm(range(1, n_epochs + 1), desc="epoch"):
            train(model, optimizer, mt_train_dl,
                  epoch=epoch,
                  spec_id_to_n_classes={spec_id: spec.n_classes for spec_id, spec in spec_id_to_specs.items()},
                  half_precision=args.amp,
                  )
            log(f"Epoch {epoch}")
            if epoch % eval_interval == 0:
                batch_num = len(mt_train_dl) * epoch  # Each batch is one update to the model.
                for spec_id, test_dataloader in spec_id_to_test_dl.items():
                    spec = spec_id_to_specs[spec_id]
                    eval_and_save(
                        model, test_dataloader, run_dir,
                        epoch=epoch, batch_num=batch_num, run=run, n_runs=n_runs, spec=spec, args=args,
                    )

                if args.save:
                    run_dir.mkdir(exist_ok=True, parents=True)
                    # Save model checkpoint (could be large!)
                    model_path = run_dir / "model.pkl"
                    torch.save(model.state_dict(), model_path)
                    log(f"Saved model to {model_path}")


def train(
        model,
        optimizer,
        multitask_dataloader: MultitaskDataLoader,
        epoch,
        *,
        spec_id_to_n_classes: Dict[str, int],
        half_precision: bool,
        log_interval=10,
) -> None:
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    scaler = amp.GradScaler()
    if half_precision:
        print("Using half precision!")

    for step, batch_list in enumerate(multitask_dataloader):
        # Zero gradient buffers
        optimizer.zero_grad()

        # Forward pass
        if half_precision:
            autocast = amp.autocast(dtype=torch.float16)
        else:
            autocast = contextlib.nullcontext()

        with autocast:
            for spec_id, batch in batch_list:
                losses = {}
                model_inputs = {k: v.cuda() for k, v in batch.items()
                                if k in data.MODEL_INPUT_KEYS}
                labels = batch["label"].cuda()

                convert_one_hot = True
                if convert_one_hot:
                    target = F.one_hot(labels, num_classes=spec_id_to_n_classes[spec_id]).cuda().float()
                else:
                    target = labels
                logits = model(spec_id, **model_inputs)
                _loss = criterion(logits, target)
                assert _loss.shape == ()
                assert spec_id not in losses.keys()
                losses[spec_id] = _loss
            total_loss = torch.mean(torch.stack([loss for loss in losses.values()]))

        # Backward pass and Update weights
        if half_precision:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        if step % log_interval == 0 and step > 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step, len(multitask_dataloader), 100. * step / len(multitask_dataloader), total_loss))


# test_dataloader
def eval_and_save(model, test_dataloader, run_dir: pathlib.Path, *,
                  epoch, batch_num, run, n_runs, spec, args,
                  ):
    test_acc, f1_micro, f1_macro, test_f1_all, precisions, recalls, logits, labels = evaluate(
        model,
        test_dataloader,
        name="test",
        spec_id=spec.id,  # Send the parameter to `forward`
    )
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
        test_f1_all=test_f1_all,
        precisions=precisions,
        recalls=recalls,
        test_acc=test_acc,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
    )

    # Write run_results
    timestamp = utils.make_unique_filename()
    run_save_path = run_dir / f"{timestamp}_run_results.pkl"
    with open(run_save_path, "wb") as f:
        pickle.dump(run_results, f)
    log(f"Dumped run_results for spec_id '{spec.id}' (epoch={epoch}, batch_num={batch_num}) to '{run_save_path}'")


def evaluate(model, test_hard_dataloader, spec_id: str, name=None):
    model.eval()
    all_predictions = []
    all_labels = []
    all_logits = []

    for batch in test_hard_dataloader:
        model_inputs = {k: v.cuda() for k, v in batch.items()
                        if k in data.MODEL_INPUT_KEYS}

        with torch.no_grad():
            logits = model(spec_id, **model_inputs)

        predictions_b: np.ndarray = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels_b: np.ndarray = batch["label"].detach().cpu().numpy()
        all_predictions.append(predictions_b)
        all_labels.append(labels_b)
        all_logits.append(logits.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    all_logits = np.concatenate(all_logits)
    acc = np.mean(all_labels == all_predictions)
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
    parser.add_argument("--nruns", "-r", type=int, default=1)
    parser.add_argument("--eval_split", default="valid", choices=["valid", "test"])

    parser.add_argument("--ngpus", "-n", type=int, default=1)
    parser.add_argument("--dry_run", "-d", action="store_true")

    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp autocasting to reduce compute/memory.")

    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--weight_decay", "-w", type=float, default=0.01)
    parser.add_argument("--learning_rate", "-l", type=float, default=2e-5)
    parser.add_argument("--save", "-S", action="store_true", help="Save checkpoints. (warning: usually large!)")

    epoch_or_updates_group = parser.add_mutually_exclusive_group()
    epoch_or_updates_group.add_argument("--nepochs", "-e", type=int, default=4)
    parser.add_argument("--eval_interval", "-i", type=int, default=1)
    parser.add_argument("--drop_additional", action="store_true")
    return parser


def console_main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    our_specs: List[maud.specs.BaseDatasetSpec]
    our_specs = auto_load_data.get_all_valid_specs(verbosity=0)
    main(our_specs[::], args)


if __name__ == "__main__":
    console_main()
