import csv
import datetime
import pathlib
import math
from typing import Iterable, List, Optional
import warnings

import randomname
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig, AdamW


def handle_none_entered_context(context: str) -> Optional[str]:
    """Returns a (cleaned) context if context is valid. If the context is null,
    then returns None."""
    if "(None entered)" in context:
        if len(context) < 75:  # Consider it invalid.
            return None
        else:  # Assume that this is a context that has spurious (None Entered)
            context = context.replace("\n\n(None entered)", "")
            context = context.replace("\n(None entered)", "")
            context = context.replace("(None entered)", "")
    return context


def make_unique_filename() -> str:
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    rand_name = randomname.get_name()
    return f"{timestamp}_{rand_name}"


def load_model(args, num_labels: int, load_path=None, cache_dir=None):
    if cache_dir is not None:
        config = AutoConfig.from_pretrained(args.model, num_labels=num_labels, cache_dir=cache_dir)
    else:
        config = AutoConfig.from_pretrained(args.model, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config)
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(args.ngpus)])
    print('\nPretrained model "{}" loaded'.format(args.model))
    return model


def load_multi_head_model(args, specs, load_path=None):
    from maud import multi_head_model
    model = multi_head_model.MultiHeadModel(base_model_name=args.model)
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))
        spec_ids = {spec.id for spec in specs}
        assert spec_ids == set(model.classification_heads.keys())
    else:
        for spec in specs:
            model.add_head_from_spec(spec)
    print(f'\nMultihead model "{args.model}" with {len(specs)} classification heads loaded')
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(args.ngpus)])
    return model


def make_optimizer(model, *, args):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    return optimizer



def strict_zip(*seqs):
    lengths = [len(seq) for seq in seqs]
    assert len(set(lengths)) == 1, f"Mismatching lengths of inputs {lengths}"
    return zip(*seqs)


class DataLoaderFixedBatches:
    """Set the number of batches"""

    def __init__(self, data_loader, batches_per_iter: int):
        self.data_loader = data_loader
        self.batches_per_iter = batches_per_iter
        self.it = None

    @property
    def dataset(self):
        return self.data_loader.dataset

    def __len__(self) -> int:
        return self.batches_per_iter

    def __iter__(self):
        if self.it is None:
            self.it = iter(self.data_loader)
        for _ in range(self.batches_per_iter):
            try:
                batch = next(self.it)
            except StopIteration:
                self.it = iter(self.data_loader)
                batch = next(self.it)
            yield batch

    def close(self):
        self.it = None


class EndlessIterator:
    """A DataLoader wrapper that repeats the DataLoader.

    Arguments:
        data_loader: The iterable.
        n_repeats: If None then repeat forever. If a positive interger,
            then repeat until the iterator is exhausted `repeat_iter` times.
    """
    def __init__(self, data_loader, n_repeats=None):
        self.data_loader = data_loader
        if n_repeats is not None:
            assert n_repeats >= 1
        else:
            n_repeats = math.inf
        self.n_repeats = n_repeats
        self.it = None
        self.new_iter_count = 0

    def __iter__(self):
        i = 0
        it = None
        new_iter = True
        while i < self.n_repeats:
            try:
                if it is None:
                    it = iter(self.data_loader)
                    new_iter = True
                yield next(it)
                new_iter = False
            except StopIteration:
                if new_iter:
                    raise RuntimeError(f"Empty iterator! {it} {self.data_loader}")
                i += 1
                it = None
