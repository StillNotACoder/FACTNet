"""Convert per-event grayscale image folders using the supplied sampling rules."""

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def prepare(input_dir, output, train_fraction=0.85, threshold=3.0, overwrite=False):
    input_dir, output = Path(input_dir), Path(output)
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one.")
    folders = sorted(path for path in input_dir.iterdir() if path.is_dir())
    boundary = int(len(folders) * train_fraction)
    if not 0 < boundary < len(folders):
        raise ValueError("At least one event folder is required in each split.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w" if overwrite else "x") as handle:
        for split, events in (("train", folders[:boundary]), ("test", folders[boundary:])):
            group = handle.create_group(split)
            count = 0
            for event in tqdm(events, desc=split):
                paths = sorted(
                    path
                    for path in event.iterdir()
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
                )
                frames = []
                for path in paths:
                    with Image.open(path) as image:
                        frames.append(np.asarray(image.convert("L")))
                index = 10
                # Preserve original strict endpoint, mean filter, and stride rules.
                while index + 20 < len(frames):
                    if frames[index].mean() > threshold:
                        group.create_dataset(
                            str(count),
                            data=frames[index - 5 : index + 20],
                            dtype="uint8",
                            compression="lzf",
                        )
                        count += 1
                        index += 15 if split == "train" else 5
                    else:
                        index += 5
            group["all_len"] = count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.85)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(args.input_dir, args.output, args.train_fraction, args.threshold, args.overwrite)


if __name__ == "__main__":
    main()
