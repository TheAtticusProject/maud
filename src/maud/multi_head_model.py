import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import torch as th
from torch.utils.data import DataLoader
import transformers
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from maud import specs, utils


class MultitaskDataLoader:
    """"""
    def __init__(
            self,
            sid_to_dl: Dict[str, Iterable[dict]],
            n_dl_batches_per_epoch: int,
            n_specs_per_batch_list: int,
    ):
        """MultitaskDataLoader interleaves between DataLoaders from different
        tasks to yield lists of batches that suitable for MAUD training.

        Arguments:
            n_dl_batches_per_epoch: The number of batches of  each
                inner data loader yielded every epoch (one for loop pass
                through __iter__(). This may be different than the number
                of batches per epoch of this DataLoader.
            n_specs_per_batch_list: The preferred number of specs that are
                grouped together into a single batch during each call to
                __iter__().
                # If this is None, then return a batch for every spec in
                # every mega-batch.
            sid_to_dl: A dictionary mapping spec IDs to respective data
                loaders. Different data loaders are allowed to have different
                lengths.
        """
        self.n_dl_batches_per_epoch = n_dl_batches_per_epoch
        assert n_specs_per_batch_list > 0
        # assert n_specs_per_batch_list < len(sid_to_dl)
        self.n_specs_per_batch_list = n_specs_per_batch_list
        self.sid_to_dl = sid_to_dl
        self.sid_to_repeated_dl = {
            k: utils.EndlessIterator(v)
            for k, v in sid_to_dl.items()
        }

    def __iter__(self) -> Iterable[List[Tuple[str, dict]]]:
        return _harder_iter(
            self.sid_to_repeated_dl,
            n_dl_batches_per_epoch=self.n_dl_batches_per_epoch,
            n_specs_per_batch_list=self.n_specs_per_batch_list,
            shuffle_specs=True,
        )

    def __len__(self) -> int:
        n_spec_groups = math.ceil(len(self.sid_to_dl) / self.n_specs_per_batch_list)
        return self.n_dl_batches_per_epoch * n_spec_groups

    def close(self):
        self.sid_to_repeated_dl = self.sid_to_dl = None


def _easy_iter(
        iter_dict: Dict[str, Iterable[dict]],
        n_dl_batches_per_epoch: int,
) -> Iterable[List[Tuple[str, dict]]]:
    """Given a dictionary mapping spec_id keys to data_loader values, yields
    structured lists with one batch from every data_loader. Each element of the
    return list is a tuple (spec_id, batch_dict).
    """
    raw_iters = {k: iter(v) for k, v in iter_dict.items()}
    for i in range(n_dl_batches_per_epoch):
        result = []
        for k, it in raw_iters.items():
            batch = next(it)
            result.append((k, batch))
        yield result


def _harder_iter(
        dl_dict: Dict[str, Iterable[dict]],
        n_dl_batches_per_epoch: int,
        n_specs_per_batch_list: int,
        shuffle_specs: bool = False,
) -> Iterable[List[Tuple[str, dict]]]:
    """Given a dictionary mapping spec_ids keys to data_loader values,
    yields structured lists of length `n_specs_per_batch_list`, where every
    element of the list represents a batch from a different data_loaders in
    roughly round-robin order. (The final yielded list may have fewer than
    `n_dl_batches_per_epochs`).
    """
    assert n_specs_per_batch_list > 0

    for i in range(n_dl_batches_per_epoch):
        raw_iters = [(k, iter(v)) for k, v in dl_dict.items()]
        if shuffle_specs:
            random.shuffle(raw_iters)
        step = n_specs_per_batch_list
        for start in range(0, len(raw_iters), step):
            iters = raw_iters[start:start+step]
            result = []
            for k, it in iters:
                try:
                    result.append((k, next(it)))
                except StopIteration:
                    raise
            yield result


def _make_classification_head(base_model_name: str, num_labels: int):
    cfg = transformers.AutoConfig.from_pretrained(base_model_name, num_labels=num_labels)
    if not hasattr(cfg, "classifier_dropout"):  # RobertaClassificationHead compatibility
        cfg.classifier_dropout = None
    classifier = RobertaClassificationHead(cfg)
    return classifier


class MultiHeadModel(th.nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        self.base_model_name = base_model_name
        self.base_model = transformers.AutoModel.from_pretrained(base_model_name)
        self.classification_heads = th.nn.ModuleDict()

    @staticmethod
    def _convert_spec_id(x: str) -> str:
        """Torch module names cannot contain '.'. Converts them to '__'."""
        return x.replace(".", "__")

    def forward(self, task_key: str, *, input_ids, attention_mask) -> th.Tensor:
        """Returns the classification logits."""
        task_key = self._convert_spec_id(task_key)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs["last_hidden_state"]
        classifier = self.classification_heads[task_key]
        return classifier(sequence_output)

    def add_head(self, task_key: str, n_labels: int) -> None:
        """Initialize a fresh classification head."""
        task_key = self._convert_spec_id(task_key)
        if task_key in self.classification_heads:
            raise ValueError(f"Duplicate task_key: {task_key}")
        head = _make_classification_head(self.base_model_name, num_labels=n_labels)
        self.classification_heads[task_key] = head

    def add_head_from_spec(self, spec: specs.BaseDatasetSpec) -> None:
        """Like self.add_head(), but use a spec instead."""
        self.add_head(task_key=spec.id, n_labels=spec.n_classes)

    def _saving_STUB(self) -> None:
        """
        Torch: Saving Multiple Models in One File
        """

