import h5py
import numpy as np
import pytest
import pickle
import pandas as pd
from PIL import Image

from datasets.get_datasets import build_dataloaders
from factnet.cli import parse_args


def make_shanghai(path):
    with h5py.File(path, "w") as handle:
        for split, count in (("train", 4), ("test", 3)):
            group = handle.create_group(split)
            group["all_len"] = count
            for i in range(count):
                group[str(i)] = np.full((4, 16, 16), 40 + i, dtype=np.uint8)


@pytest.mark.parametrize("workers", [0, 2])
def test_test_split_and_tail_batch(tmp_path, workers):
    path = tmp_path / "radar.h5"
    make_shanghai(path)
    train, val, test, _, scale, thresholds = build_dataloaders(
        "shanghai",
        data_path=path,
        img_size=16,
        seq_len=4,
        batch_size=2,
        splits=("test",),
        num_workers=workers,
    )
    assert train is val is None
    batches = list(test)
    assert [len(batch) for batch in batches] == [2, 1]
    assert batches[0].shape == (2, 4, 1, 16, 16)
    assert float(batches[0][0, 0, 0, 0, 0]) == pytest.approx(40 / 255)
    assert scale == 90 and list(thresholds) == [20, 30, 35, 40]


def test_cikm_protocol_and_config_override(tmp_path):
    args = parse_args(["--dataset", "cikm", "--smoke-test"])
    assert args.frames_out == 10
    config = tmp_path / "config.yaml"
    config.write_text("dataset: cikm\nbatch_size: 4\n", encoding="utf-8")
    args = parse_args(["--config", str(config), "--batch-size", "2", "--smoke-test"])
    assert args.batch_size == 2 and args.frames_out == 10


def test_test_only_hdf5_does_not_require_training_split(tmp_path):
    path = tmp_path / "test-only.h5"
    with h5py.File(path, "w") as handle:
        handle["test_len"] = 1
        handle.create_group("test")["sample_1"] = np.full((15, 101, 101), 255, dtype=np.uint8)
    *_, loader, _, scale, _ = build_dataloaders(
        "cikm", data_path=path, img_size=128, batch_size=4, splits=("test",)
    )
    frames = next(iter(loader))
    assert frames.shape == (1, 15, 1, 128, 128)
    assert frames[0, 0, 0, 64, 64] == 1
    assert frames[0, 0, 0, 0, 0] == 0  # Preserve the original crop/padding behavior.
    assert scale == 80


@pytest.mark.parametrize("workers", [0, 2])
def test_sevir_windows_and_spawn_serialization(tmp_path, workers):
    root = tmp_path / "sevir"
    (root / "data").mkdir(parents=True)
    with h5py.File(root / "data" / "vil.h5", "w") as handle:
        handle["vil"] = np.full((1, 16, 16, 49), 255, dtype=np.int16)
    pd.DataFrame(
        [
            {
                "id": "event-test",
                "file_name": "vil.h5",
                "file_index": 0,
                "img_type": "vil",
                "time_utc": "2019-07-01 00:00:00",
                "pct_missing": 0,
            }
        ]
    ).to_csv(root / "CATALOG.csv", index=False)
    _, _, loader, _, scale, _ = build_dataloaders(
        "sevir",
        data_path=root,
        seq_len=25,
        img_size=16,
        stride=13,
        batch_size=4,
        splits=("test",),
        num_workers=workers,
    )
    # One 49-frame event yields starts 0 and 13, both kept in a tail batch.
    assert len(loader.dataset) == 2
    batch = next(iter(loader))
    assert batch.shape == (2, 25, 1, 16, 16)
    assert float(batch.min()) == pytest.approx(1)
    # The dataset has now opened HDF5; serialization must not retain the handle.
    restored = pickle.loads(pickle.dumps(loader.dataset))
    assert restored[0].shape == (1, 25, 1, 16, 16)
    loader.dataset.sevir_dataloader.close()
    restored.sevir_dataloader.close()
    assert scale == 255


def test_shanghai_preparation_preserves_window_sampling(tmp_path):
    from scripts.prepare_shanghai import prepare

    events = tmp_path / "events"
    for name in ("event-a", "event-b"):
        folder = events / name
        folder.mkdir(parents=True)
        for index in range(40):
            Image.fromarray(np.full((4, 4), index, dtype=np.uint8)).save(
                folder / f"{index:03d}.png"
            )
    output = tmp_path / "prepared.h5"
    prepare(events, output, train_fraction=0.5)
    with h5py.File(output, "r") as handle:
        assert handle["train"]["all_len"][()] == 1
        assert handle["test"]["all_len"][()] == 2
        assert handle["train"]["0"].shape == (25, 4, 4)
        assert handle["train"]["0"][0, 0, 0] == 5
        assert handle["test"]["1"][0, 0, 0] == 10
