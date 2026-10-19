import torch
import pytest

from models.factnet import get_model


def small_model():
    return get_model(dim=8, T_in=2, T_out=3, input_shape=(16, 16), n_layers=1, ie_layers=1)


def test_backward_and_old_state_dict():
    torch.manual_seed(7)
    model = small_model()
    inputs, target = torch.rand(2, 2, 1, 16, 16), torch.rand(2, 3, 1, 16, 16)
    prediction, losses = model(inputs, target, compute_loss=True, training_step=0)
    assert prediction.shape == target.shape
    losses["total_loss"].backward()
    for branch in (model.intensity_branch, model.motion_branch, model.reconstruction):
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in branch.parameters())
    old_state = {key: value for key, value in model.state_dict().items() if key != "training_steps"}
    restored = small_model()
    restored.load_state_dict(old_state, strict=True)
    with torch.no_grad():
        expected, _ = model(inputs)
        actual, _ = restored(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_validation_does_not_advance_amplitude_decay():
    model = small_model()
    inputs, target = torch.rand(1, 2, 1, 16, 16), torch.rand(1, 3, 1, 16, 16)
    _, before = model(inputs, target, compute_loss=True, training_step=2)
    steps = model.training_steps.clone()
    model.eval()
    with torch.no_grad():
        model(inputs)
        model(inputs)
    torch.testing.assert_close(model.training_steps, steps)
    _, after = model(inputs, target, compute_loss=True, training_step=2)
    torch.testing.assert_close(before["ampli_loss"], after["ampli_loss"])


def test_invalid_input_is_reported():
    with pytest.raises(ValueError, match="even"):
        get_model(input_shape=(15, 16))
    with pytest.raises(ValueError, match="Expected"):
        small_model()(torch.rand(1, 5, 1, 16, 16))
