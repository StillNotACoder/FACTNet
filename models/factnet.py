"""FACTNet assembly, compatible parameter names, and training loss."""

import torch
from torch import nn
from .factnet_modules.intensity import IENet
from .factnet_modules.motion import MDNet
from .factnet_modules.reconstruction import SpatialFrequencyReconstruction


class FACTNet(nn.Module):
    """Input/output: [batch, time, channel, height, width].

    Old attribute keys are retained for checkpoint compatibility.
    See docs/architecture.md for differences from the manuscript.
    """

    def __init__(
        self,
        pre_seq_length=5,
        aft_seq_length=20,
        input_shape=(128, 128),
        input_dim=1,
        hidden_dim=64,
        n_layers=3,
        spec_num=20,
        kernel_size=1,
        pha_weight=0.01,
        anet_weight=0.1,
        amp_weight=0.01,
        aweight_stop_steps=5000,
        ie_layers=3,
    ):
        super().__init__()
        if (
            min(
                pre_seq_length,
                aft_seq_length,
                *input_shape,
                input_dim,
                hidden_dim,
                n_layers,
                ie_layers,
                spec_num,
            )
            <= 0
        ):
            raise ValueError("Sequence lengths, dimensions, depths, and spec_num must be positive.")
        if any(s % 2 for s in input_shape):
            raise ValueError("The supplied model requires even spatial sizes.")
        if hidden_dim % 8 or (hidden_dim > 32 and hidden_dim % 32):
            raise ValueError("hidden_dim must satisfy GroupNorm groups (e.g. 8,16,32,64).")
        if aweight_stop_steps <= 0 or min(pha_weight, anet_weight, amp_weight) < 0:
            raise ValueError("Loss weights must be nonnegative and decay steps positive.")
        self.amplinet = IENet(
            pre_seq_length, aft_seq_length, input_dim, hidden_dim, n_layers=ie_layers
        )
        self.phasenet = MDNet(
            input_shape,
            pre_seq_length,
            aft_seq_length,
            input_dim,
            hidden_dim,
            n_layers,
            kernel_size,
        )
        self.alphamixer = SpatialFrequencyReconstruction(
            input_shape, spec_num, input_dim, hidden_dim, aft_seq_length
        )
        self.input_shape = tuple(input_shape)
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.pre_seq_length, self.aft_seq_length = pre_seq_length, aft_seq_length
        self.spec_num = spec_num
        self.pha_weight, self.anet_weight, self.amp_weight = pha_weight, anet_weight, amp_weight
        self.aweight_stop_steps = aweight_stop_steps
        self.criterion = nn.MSELoss()
        h, w = input_shape
        mask = torch.zeros(h, w // 2 + 1)
        mask[..., :spec_num, :spec_num] = 1.0
        mask[..., -spec_num:, :spec_num] = 1.0
        self.register_buffer("spec_mask", mask)
        self.register_buffer("training_steps", torch.zeros((), dtype=torch.long))

    @property
    def intensity_branch(self):
        return self.amplinet

    @property
    def motion_branch(self):
        return self.phasenet

    @property
    def reconstruction(self):
        return self.alphamixer

    def forward(self, frames_in, frames_gt=None, compute_loss=False, training_step=None):
        expected = (self.pre_seq_length, self.input_dim, *self.input_shape)
        if frames_in.ndim != 5 or tuple(frames_in.shape[1:]) != expected:
            raise ValueError(f"Expected [B,{expected}], got {tuple(frames_in.shape)}.")
        intensity = torch.sigmoid(self.amplinet(frames_in))
        motion, phase, _ = self.phasenet(frames_in)
        pred = self.alphamixer(intensity, motion, phase)
        if not compute_loss:
            return pred, None
        if frames_gt is None or frames_gt.shape != pred.shape:
            raise ValueError("Loss computation requires matching target frames.")
        step = int(self.training_steps.item()) if training_step is None else int(training_step)
        if step < 0:
            raise ValueError("training_step must be nonnegative.")
        # Match the original first-update weight; evaluation no longer advances it.
        amplitude_weight = self.amp_weight * max(0.0, 1.0 - (step + 1) / self.aweight_stop_steps)
        target_fft = torch.fft.rfft2(frames_gt)
        # Retain the original operation order for numerical compatibility.
        phase_loss = (
            1 - torch.cos(torch.angle(target_fft) * self.spec_mask - phase * self.spec_mask)
        ).sum()
        phase_loss = phase_loss / (
            self.spec_mask.sum() * frames_in.shape[0] * self.aft_seq_length * self.input_dim
        )
        amp_loss = self.criterion(torch.abs(torch.fft.rfft2(intensity)), torch.abs(target_fft))
        ie_loss = self.criterion(intensity, frames_gt)
        loss = (
            self.criterion(pred, frames_gt)
            + self.pha_weight * phase_loss
            + amplitude_weight * amp_loss
            + self.anet_weight * ie_loss
        )
        self.training_steps.fill_(step + 1)
        return pred, {
            "total_loss": loss,
            "phase_loss": self.pha_weight * phase_loss,
            "ampli_loss": amplitude_weight * amp_loss,
            "anet_loss": self.anet_weight * ie_loss,
        }

    def predict(self, frames_in, frames_gt=None, compute_loss=False, **kwargs):
        """Compatibility wrapper; call the model itself for distributed training."""
        return self(frames_in, frames_gt, compute_loss, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = state_dict.copy()
        state_dict.setdefault("training_steps", self.training_steps.detach().clone())
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def get_model(
    img_channels=1,
    dim=64,
    T_in=5,
    T_out=20,
    input_shape=(128, 128),
    n_layers=3,
    spec_num=20,
    pha_weight=0.01,
    anet_weight=0.1,
    amp_weight=0.01,
    aweight_stop_steps=5000,
    ie_layers=3,
    kernel_size=1,
):
    """n_layers controls MD depth; IE defaults to the original three stages."""
    return FACTNet(
        pre_seq_length=T_in,
        aft_seq_length=T_out,
        input_shape=input_shape,
        input_dim=img_channels,
        hidden_dim=dim,
        n_layers=n_layers,
        spec_num=spec_num,
        kernel_size=kernel_size,
        pha_weight=pha_weight,
        anet_weight=anet_weight,
        amp_weight=amp_weight,
        aweight_stop_steps=aweight_stop_steps,
        ie_layers=ie_layers,
    )
