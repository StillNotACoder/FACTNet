"""Path-configurable factories for the four published benchmarks."""

import datetime
from functools import partial
from pathlib import Path


def vis_res(*args, **kwargs):
    """Compatibility import; plotting dependencies are loaded on demand."""
    from utils.visualization import vis_res as render

    return render(*args, **kwargs)


def normalize_dataset_name(name):
    name = name.lower()
    if name == "meteonet":
        name = "meteo"
    if name not in {"cikm", "shanghai", "meteo", "sevir"}:
        raise ValueError(
            f"Unknown dataset {name!r}; choose cikm, shanghai, meteo/meteonet, or sevir."
        )
    return name


def _normalize_splits(splits):
    if isinstance(splits, str):
        splits = (splits,)
    result = tuple("val" if split == "valid" else split for split in splits)
    if not result or any(split not in {"train", "val", "test"} for split in result):
        raise ValueError("splits must contain train, val/valid, and/or test.")
    return set(result)


def get_dataset(
    data_name,
    img_size=128,
    seq_len=None,
    *,
    data_path=None,
    splits=("train", "val", "test"),
    batch_size=1,
    stride=13,
    eval_batch_size=None,
    seed=1,
    **kwargs,
):
    """Return ``(train, val, test, color_fn, pixel_scale, thresholds)``.

    ``data_path`` is an HDF5 file for CIKM, Shanghai, or MeteoNet, and a
    directory containing ``CATALOG.csv`` and ``data/`` for SEVIR. Use
    ``splits=("test",)`` to evaluate without loading training/validation data;
    unrequested splits are returned as ``None``. CIKM defaults to 15 frames,
    and the other datasets default to 25 frames.

    SEVIR keeps its original prebatched Dataset interface. Use its
    ``get_torch_dataloader`` method or ``build_dataloaders`` below.
    """
    if kwargs:
        raise TypeError(f"Unexpected dataset options: {', '.join(sorted(kwargs))}")
    name = normalize_dataset_name(data_name)
    requested = _normalize_splits(splits)
    seq_len = (15 if name == "cikm" else 25) if seq_len is None else seq_len
    if img_size <= 0 or seq_len <= 0 or batch_size <= 0:
        raise ValueError("img_size, seq_len, and batch_size must be positive.")
    if name == "cikm" and seq_len > 15:
        raise ValueError("CIKM contains 15 frames per sample; use seq_len <= 15 (normally 5+10).")
    if name == "sevir" and (stride <= 0 or seq_len > 49):
        raise ValueError("SEVIR requires stride > 0 and 0 < seq_len <= 49.")
    if data_path is None:
        raise ValueError("data_path is required: an HDF5 file, or the SEVIR root directory.")
    path = Path(data_path).expanduser()
    if name == "sevir":
        if not path.is_dir():
            raise FileNotFoundError(f"SEVIR root directory does not exist: {path}")
    elif not path.is_file():
        raise FileNotFoundError(f"Dataset HDF5 file does not exist: {path}")

    if name == "cikm":
        from .dataset_cikm import CIKM as dataset_class, gray2color, PIXEL_SCALE, THRESHOLDS
    elif name == "shanghai":
        from .dataset_shanghai import Shanghai as dataset_class, gray2color, PIXEL_SCALE, THRESHOLDS
    elif name == "meteo":
        from .dataset_meteonet import Meteo as dataset_class, gray2color, PIXEL_SCALE, THRESHOLDS
    else:
        from .dataset_sevir import SEVIRTorchDataset, gray2color, PIXEL_SCALE, THRESHOLDS

    datasets = {"train": None, "val": None, "test": None}
    if name == "sevir":
        # Preserve the original date boundaries and strict/inclusive filters.
        boundaries = {
            "train": (None, datetime.datetime(2019, 1, 1)),
            "val": (datetime.datetime(2019, 1, 1), datetime.datetime(2019, 6, 1)),
            "test": (datetime.datetime(2019, 6, 1), datetime.datetime(2019, 12, 31)),
        }
        eval_batch_size = batch_size * 2 if eval_batch_size is None else eval_batch_size
        if eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive.")
        for split in requested:
            start, end = boundaries[split]
            datasets[split] = SEVIRTorchDataset(
                dataset_dir=str(path),
                img_size=img_size,
                seq_len=seq_len,
                stride=stride,
                batch_size=batch_size if split == "train" else eval_batch_size,
                split_mode="uneven",
                sample_mode="sequent",
                shuffle=split == "train",
                shuffle_seed=seed,
                num_shard=1,
                rank=0,
                start_date=start,
                end_date=end,
                preprocess=True,
                rescale_method="01",
                verbose=False,
            )
    else:
        for split in requested:
            datasets[split] = dataset_class(
                str(path),
                img_size=img_size,
                type=split,
                seq_len=seq_len,
            )

    color_fn = partial(
        vis_res, pixel_scale=PIXEL_SCALE, thresholds=THRESHOLDS, gray2color=gray2color
    )
    return datasets["train"], datasets["val"], datasets["test"], color_fn, PIXEL_SCALE, THRESHOLDS


def _collate_sevir(samples):
    import torch

    return torch.cat(samples, dim=0)


def build_dataloaders(
    data_name,
    img_size=128,
    seq_len=None,
    *,
    data_path=None,
    splits=("train", "val", "test"),
    batch_size=4,
    eval_batch_size=None,
    num_workers=0,
    stride=13,
    pin_memory=False,
    seed=1,
):
    """Return the same six-tuple with DataLoaders in the first three slots.

    Every batch has shape ``[B,T,1,H,W]``. SEVIR internally loads one window
    per Dataset item, so ``DataLoader.batch_size`` describes real samples.
    Validation and test include the last incomplete batch without padding;
    training drops its incomplete last batch.
    """
    import torch
    from torch.utils.data import DataLoader

    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive; num_workers must be nonnegative.")
    name = normalize_dataset_name(data_name)
    eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive.")
    datasets = get_dataset(
        name,
        img_size,
        seq_len,
        data_path=data_path,
        splits=splits,
        batch_size=1,
        eval_batch_size=1,
        stride=stride,
        seed=seed,
    )
    loaders = []
    for split, dataset in zip(("train", "val", "test"), datasets[:3]):
        if dataset is None:
            loaders.append(None)
        else:
            loaders.append(
                DataLoader(
                    dataset,
                    batch_size=batch_size if split == "train" else eval_batch_size,
                    shuffle=split == "train",
                    num_workers=num_workers,
                    drop_last=split == "train",
                    pin_memory=pin_memory,
                    collate_fn=_collate_sevir if name == "sevir" else None,
                    generator=torch.Generator().manual_seed(seed),
                )
            )
    return (*loaders, *datasets[3:])
