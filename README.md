# FACTNet

Code for **FACTNet: Disentangling Motion and Intensity in Spatial-Frequency Domains for Precipitation Nowcasting**.

FACTNet combines an intensity evolution branch (IENet), a motion dynamics branch (MDNet), and spatial-frequency reconstruction for radar echo nowcasting.

## Installation

Use Python 3.10 or newer. The project targets PyTorch 2.7.1 / torchvision 0.22.1.

```bash
conda create -n factnet python=3.10 -y
conda activate factnet

# Install the PyTorch build appropriate for your machine first.
# Example for the CUDA 12.8 environment used during training:
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

For CPU checks, use the PyTorch CPU wheel index instead. An alternative environment definition is provided in `env.yaml`; run Conda commands from the repository root.

## Repository structure

```text
run.py                         # Thin CLI entry point
factnet/
  cli.py                       # Arguments and YAML configuration
  runner.py                    # Training, evaluation, full checkpoints
models/
  factnet.py                   # FACTNet assembly and losses
  factnet_modules/
    intensity.py               # IENet, IEBlock, FreMLP, local spatial branch
    motion.py                  # MDNet, MSCM, FGA, MSK, spatial guidance
    reconstruction.py          # Fourier reconstruction and learned fusion
    common.py                  # Encoder/decoder and convolutional blocks
    layers.py                  # Required original normalization/DW layers
datasets/                      # Four benchmark adapters
utils/                         # Metrics and optional visualization
configs/                       # Dataset training configurations
tests/                         # Synthetic model/data/metric checks
docs/                          # Architecture, data, and metric details
```

The refactor preserves the supplied model computations and existing parameter keys. Some operations in this source differ from the manuscript equations: the current motion routing uses spatial filters, and reconstruction uses learned convolutional fusion. The complete correspondence is documented in [docs/architecture.md](docs/architecture.md).

## Data

Provide an explicit local path; no personal paths are embedded in the source.

| Dataset | `--data-path` | Input / output frames | Evaluation scale |
|---|---|---|---|
| SEVIR | Root containing `CATALOG.csv` and `data/` | 5 / 20 | 255 |
| MeteoNet (`meteo`) | Prepared HDF5 file | 5 / 20 | 90 |
| Shanghai Radar | Prepared HDF5 file | 5 / 20 | 90 |
| CIKM | Prepared HDF5 file | 5 / 10 | 80 |

HDF5 layouts, original preprocessing, split mappings, and threshold sets are described in [docs/data.md](docs/data.md). The supplied Shanghai/MeteoNet adapters map validation to the test group, and CIKM validation uses the first 1000 test entries. These existing mappings are retained and explicitly recorded; consider them when interpreting checkpoint selection and test results.

Paths can also be provided through `FACTNET_DATA_SEVIR`, `FACTNET_DATA_METEO`, `FACTNET_DATA_SHANGHAI`, or `FACTNET_DATA_CIKM`.

The supplied Shanghai image-folder conversion is available as a standalone command:

```bash
python scripts/prepare_shanghai.py --input-dir /path/to/event_folders --output /path/to/shanghai.h5
```

## Training

```bash
python run.py --config configs/shanghai.yaml --data-path /path/to/shanghai.h5 --output-dir outputs/shanghai
python run.py --config configs/meteo.yaml --data-path /path/to/meteo_radar.h5 --output-dir outputs/meteo
python run.py --config configs/sevir.yaml --data-path /path/to/SEVIR --output-dir outputs/sevir
python run.py --config configs/cikm.yaml --data-path /path/to/cikm.h5 --output-dir outputs/cikm
```

CLI arguments override YAML values. Defaults are AdamW, 100 epochs, batch size 4 per process, learning rate 1e-4, 1000 warmup updates, and cosine decay. `--grad-acc-step` performs actual gradient accumulation. Effective batch size is `batch_size * grad_acc_step * number_of_processes`.

Each experiment saves its configuration, logs, full `checkpoints/last` state, and `checkpoints/best` selected by validation CSI. Full checkpoints include model, optimizer, scheduler, EMA (when enabled), scaler/RNG states managed by Accelerate, and progress metadata.

```bash
# Resume at a completed epoch boundary with the same training configuration.
python run.py --config configs/shanghai.yaml --data-path /path/to/shanghai.h5 --resume outputs/shanghai/checkpoints/last

# Initialize a new run from weights rather than resuming optimizer state.
python run.py --config configs/shanghai.yaml --data-path /path/to/shanghai.h5 --checkpoint /path/to/model.pt --output-dir outputs/new_run
```

`--resume` expects a directory created by this runner. Old `.pt` dictionaries with a `model` entry are supported through `--checkpoint`. Old snapshots do not contain enough state for an exact continuation of their original training process.

Optional distributed launch:

```bash
accelerate launch --num_processes 2 run.py --config configs/shanghai.yaml --data-path /path/to/shanghai.h5 --output-dir outputs/shanghai_ddp
```

Weights & Biases is disabled by default. Install `wandb` and pass `--wandb online` or `--wandb offline` to enable it. Credentials come from your own environment/login.

## Evaluation

```bash
python run.py --mode eval --config configs/shanghai.yaml --data-path /path/to/shanghai.h5 --checkpoint outputs/shanghai/checkpoints/best/model.pt --output-dir outputs/shanghai_eval

# Optional images; at most eight sequences are saved by default.
python run.py --mode eval --config configs/cikm.yaml --data-path /path/to/cikm.h5 --checkpoint /path/to/model.pt --visualize
```

Evaluation opens only the test split. Use model-size/sequence arguments matching the checkpoint. Raw weights are evaluated by default; pass `--use-ema` for saved EMA weights. Metrics are written to `test/<checkpoint-name>/eval_results.json`, including per-threshold and per-lead-time values.

CSI, HSS, SSIM, MSE, MAE, and RMSE require no pretrained image network or weight download. See [docs/metrics.md](docs/metrics.md) for exact aggregation. Validation/test retain incomplete last batches; distributed evaluation removes sampler padding with `gather_for_metrics`.

## Quick checks

```bash
# No radar files or GPU needed:
python run.py --smoke-test --device cpu --img-size 16 --dim 8 --frames-in 2 --frames-out 3 --layers 1 --ie-layers 1

pip install -r requirements-dev.txt
python -m pytest -q
```

Synthetic checks verify execution and interfaces; they do not reproduce reported benchmark scores.

## Acknowledgements

Parts of the supplied implementation were adapted from [AlphaPre](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_AlphaPre_Amplitude-Phase_Disentanglement_Model_for_Precipitation_Nowcasting_CVPR_2025_paper.html), [DiffCast](https://github.com/DeminYu98/DiffCast), [OpenSTL](https://github.com/chengtan9907/OpenSTL), and the PFGNet building blocks supplied with this project. The SEVIR loader retains its original MIT source attribution. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Radar datasets and pretrained checkpoints are not bundled. Existing local experiment images are excluded by `.gitignore`.
