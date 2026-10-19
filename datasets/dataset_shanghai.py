"""Dataset loader and original evaluation colors."""

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib import colors


class Shanghai(Dataset):
    """HDF5 radar sequences with the supplied normalization and split mapping."""

    def __init__(self, data_path, img_size=128, type="train", trans=None, seq_len=None):
        if type not in {"train", "test", "val", "valid"}:
            raise ValueError("split must be train, val/valid, or test")
        self.data_path = str(data_path)
        self.type = "test" if type in {"val", "valid"} else type
        self.seq_len = seq_len if seq_len is not None and seq_len > 0 else None
        self.transform = trans or transforms.Resize((img_size, img_size))
        with h5py.File(self.data_path, "r") as handle:
            self.all_len = int(handle[self.type]["all_len"][()])

    def __len__(self):
        return self.all_len

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        # Opening within __getitem__ avoids sharing an HDF5 handle across workers.
        with h5py.File(self.data_path, "r") as handle:
            frames = torch.from_numpy(handle[self.type][str(index)][()]).float().squeeze()
        if frames.ndim != 3:
            raise ValueError(f"Expected [T,H,W] radar sample, got {tuple(frames.shape)}")
        if self.seq_len is not None:
            if frames.shape[0] < self.seq_len:
                raise ValueError("Radar sample contains fewer frames than requested.")
            frames = frames[: self.seq_len]
        frames = frames / 255.0
        return self.transform(frames).unsqueeze(1)


PIXEL_SCALE = 90.0

COLOR_MAP = (
    np.array(
        [
            [0, 0, 0, 0],
            [0, 236, 236, 255],
            [1, 160, 246, 255],
            [1, 0, 246, 255],
            [0, 239, 0, 255],
            [0, 200, 0, 255],
            [0, 144, 0, 255],
            [255, 255, 0, 255],
            [231, 192, 0, 255],
            [255, 144, 2, 255],
            [255, 0, 0, 255],
            [166, 0, 0, 255],
            [101, 0, 0, 255],
            [255, 0, 255, 255],
            [153, 85, 201, 255],
            [255, 255, 255, 255],
        ]
    )
    / 255
)

BOUNDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, PIXEL_SCALE]

THRESHOLDS = [20, 30, 35, 40]

HMF_COLORS = np.array([[82, 82, 82], [252, 141, 89], [255, 255, 191], [145, 191, 219]]) / 255


def gray2color(image, **kwargs):

    # 定义颜色映射和边界
    cmap = colors.ListedColormap(COLOR_MAP)
    bounds = BOUNDS
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # 将图像进行染色
    colored_image = cmap(norm(image))

    return colored_image
