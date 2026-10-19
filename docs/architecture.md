# Architecture and source correspondence

This release reorganizes the implementation supplied by the author. It preserves its numerical forward computation instead of replacing it with untrained modules inferred from manuscript prose.

| Earlier code | Current class | Function |
|---|---|---|
| AlphaPre | FACTNet | Model assembly and loss |
| AmpliNet | IENet | Intensity evolution |
| AmpCell | IEBlock | Complex temporal mixing and spatial refinement |
| AmpTimeCell | FreMLP | Two complex temporal maps and a temporal residual |
| Local_CNN_Branch | LocalSpatialBranch | Local per-frame convolution |
| PhaseNet | MDNet | Encoder/evolution/decoder motion prediction |
| MSInit | MSCM | Projected multi-scale concatenation, normalization, GELU |
| PFGA.Branch | MSK | Factorized depthwise kernel and center subtraction |
| Fixed filter routing | SpatialGuidanceUnit | LoG, Scharr, and Sobel gates |
| PFGA | FGA | Guided multi-scale aggregation |
| PFG | MotionEvolutionBlock | Residual token/channel mixing |
| MidPFG | MotionEvolutionStage | MSCM followed by evolution blocks |
| AlphaMixer | SpatialFrequencyReconstruction | Fourier reconstruction and learned fusion |

## Differences from the manuscript description

- The IE path mixes complex real/imaginary components. It has no separate amplitude-only AE block with two 1x1 convolutions and a fixed input phase.
- Guidance uses fixed LoG/Scharr/Sobel filters, independent per-channel sigmoid gates, and no gradient through gate extraction. It does not implement FFT Gaussian high-pass extraction, channel mean/max projection, or across-scale softmax.
- Center suppression uses channel-wise tanh(beta), preserving the supplied code; it is not sigmoid(beta).
- Motion phase comes directly from the FFT of predicted motion frames. No historical-phase residual or UV-grid correction is present.
- Reconstruction applies a spectral mask, concatenates intensity, motion, and Fourier-reconstructed sequences, and uses learned residual convolutional fusion. The output is not a three-way sum.
- IE depth defaults to three stages independently of MD depth. The original layers option controlled only MD depth; ie_layers now makes this distinction explicit.

Required GRN and depthwise layers are copied from the matching dependency supplied by the author. Unused model-zoo layers were removed.

## Checkpoints and training-state changes

Classes now follow FACTNet terminology. Registered paths such as amplinet, phasenet, mid_pfg, and alphamixer are retained for old state-dictionary compatibility. Public intensity_branch, motion_branch, and reconstruction properties expose the new terminology without registering modules twice.

One new persistent buffer, training_steps, records loss annealing. Old state dictionaries without it load with zero. The first-update weight and uninterrupted update schedule match the original formula, subject to floating-point rounding. Validation/inference no longer advances the schedule, and all microbatches of one update use the same weight. This corrects training-state handling and can change a rerun relative to the old runner, whose counter advanced during evaluation.

Full resume uses the new runner's epoch-boundary checkpoints with matching configuration. Legacy checkpoints support strict weight loading but cannot restore missing RNG/annealing/best-score state. Arbitrary full-model pickle objects are not loaded.

Forward contract:

```python
prediction, losses = model(frames_in, frames_gt, compute_loss=True, training_step=update)
prediction, _ = model(frames_in)
```

Distributed training calls the wrapped model's forward; predict is an ordinary-inference compatibility wrapper.
