"""Optional forecast images/GIFs; dataset imports do not load plotting tools."""

from pathlib import Path

import numpy as np


HMF_COLORS = (
    np.array(
        [
            [82, 82, 82],
            [252, 141, 89],
            [255, 255, 191],
            [145, 191, 219],
        ]
    )
    / 255.0
)


def _as_sequence(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim != 4 or value.shape[1] != 1 or not len(value):
        raise ValueError("Visualization expects nonempty [T,1,H,W] arrays or tensors.")
    return np.clip(value[:, 0], 0.0, 1.0)


def vis_res(
    pred_seq,
    gt_seq,
    save_path,
    data_type="vil",
    save_grays=False,
    do_hmf=False,
    save_colored=False,
    save_gif=False,
    pixel_scale=None,
    thresholds=None,
    gray2color=None,
):
    """Save forecast/target grids, optional frame images, and optional GIFs.

    GIF writing uses Pillow, so MoviePy is not a training or plotting dependency.
    The colormap and evaluation scale are supplied by ``get_dataset``.
    """
    from matplotlib import colors
    import matplotlib.pyplot as plt

    pred_seq, gt_seq = _as_sequence(pred_seq), _as_sequence(gt_seq)
    if pred_seq.shape != gt_seq.shape:
        raise ValueError("Prediction and target sequences must have the same shape.")
    if gray2color is None or pixel_scale is None:
        raise ValueError("pixel_scale and gray2color are required.")
    output = Path(save_path)
    output.mkdir(parents=True, exist_ok=True)
    if save_grays:
        for name, sequence in (("pred", pred_seq), ("targets", gt_seq)):
            (output / name).mkdir(exist_ok=True)
            for index, frame in enumerate(sequence):
                plt.imsave(output / name / f"{index}.png", frame, cmap="gray", vmin=0, vmax=1)

    if data_type == "vil":
        pred_seq = (pred_seq * pixel_scale).astype(np.uint8)
        gt_seq = (gt_seq * pixel_scale).astype(np.uint8)
    colored_pred = np.asarray([gray2color(frame, data_type=data_type) for frame in pred_seq])
    colored_gt = np.asarray([gray2color(frame, data_type=data_type) for frame in gt_seq])
    if save_colored:
        for name, sequence in (("pred_colored", colored_pred), ("targets_colored", colored_gt)):
            (output / name).mkdir(exist_ok=True)
            for index, frame in enumerate(sequence):
                plt.imsave(output / name / f"{index}.png", frame, dpi=900)
    grid_pred = np.concatenate(list(colored_pred), axis=1)
    grid_gt = np.concatenate(list(colored_gt), axis=1)
    plt.imsave(output / "all.png", np.concatenate((grid_pred, grid_gt), axis=0))

    if save_gif:
        from PIL import Image

        for name, sequence in (("pred", colored_pred), ("targets", colored_gt)):
            images = [
                Image.fromarray((frame[..., :3] * 255).astype(np.uint8)) for frame in sequence
            ]
            images[0].save(
                output / f"{name}.gif",
                save_all=True,
                append_images=images[1:],
                duration=250,
                loop=0,
            )

    if do_hmf:
        if thresholds is None or len(thresholds) < 3:
            raise ValueError(
                "At least three thresholds are needed for the hit/miss/false-alarm plot."
            )
        pred = np.concatenate(list(pred_seq), axis=1) >= thresholds[2]
        truth = np.concatenate(list(gt_seq), axis=1) >= thresholds[2]
        # 1: correct negative; 2: false alarm; 3: miss; 4: hit.
        mask = np.ones(pred.shape, dtype=np.uint8)
        mask[pred & ~truth] = 2
        mask[~pred & truth] = 3
        mask[pred & truth] = 4
        plt.imsave(output / "hmf.png", mask, cmap=colors.ListedColormap(HMF_COLORS), vmin=1, vmax=4)
