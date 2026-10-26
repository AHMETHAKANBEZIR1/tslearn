"""Autograd coverage for caller-defined soft-DTW distance matrices."""

import numpy as np
import pytest

from tslearn.metrics import SoftDTWLossPyTorch, compute_mask
from tslearn.metrics.soft_dtw_loss_pytorch import _SoftDTWLossPyTorch

torch = pytest.importorskip("torch")


def _autograd_soft_dtw(x, y, gamma, constraint):
    """Differentiable reference recurrence omitting unreachable cells."""
    outputs = []
    for x_row, y_row in zip(x, y):
        distance = torch.cdist(x_row, y_row)
        m, n = distance.shape
        mask = np.asarray(
            compute_mask(
                np.zeros((m, 1)),
                np.zeros((n, 1)),
                global_constraint={None: 0, "itakura": 1, "sakoe_chiba": 2}[
                    constraint
                ],
                sakoe_chiba_radius=1 if constraint == "sakoe_chiba" else None,
            )
        )
        rows = [[None for _ in range(n + 1)] for _ in range(m + 1)]
        rows[0][0] = distance.new_zeros(())
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if not mask[i - 1, j - 1]:
                    continue
                predecessors = [
                    value
                    for value in (
                        rows[i - 1][j],
                        rows[i][j - 1],
                        rows[i - 1][j - 1],
                    )
                    if value is not None
                ]
                if predecessors:
                    rows[i][j] = distance[
                        i - 1, j - 1
                    ] - gamma * torch.logsumexp(
                        -torch.stack(predecessors) / gamma, dim=0
                    )
        outputs.append(rows[m][n])
    return torch.stack(outputs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("constraint", [None, "sakoe_chiba", "itakura"])
def test_soft_dtw_cdist_backward_matches_autograd(
    dtype, normalize, constraint
):
    """Custom costs support backward for constrained and free paths."""
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(
        2, 4, 2, generator=generator, dtype=dtype, requires_grad=True
    )
    y = torch.randn(
        2, 5, 2, generator=generator, dtype=dtype, requires_grad=True
    )
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = y.detach().clone().requires_grad_(True)
    gamma = 0.8
    criterion = SoftDTWLossPyTorch(
        gamma=gamma,
        normalize=normalize,
        dist_func=torch.cdist,
        global_constraint=constraint,
        sakoe_chiba_radius=1 if constraint == "sakoe_chiba" else None,
    )
    actual = criterion(x, y)
    expected = _autograd_soft_dtw(x_ref, y_ref, gamma, constraint)
    if normalize:
        expected = expected - 0.5 * (
            _autograd_soft_dtw(x_ref, x_ref, gamma, constraint)
            + _autograd_soft_dtw(y_ref, y_ref, gamma, constraint)
        )
    upstream = actual.new_tensor([0.7, -1.2])
    actual_grads = torch.autograd.grad(actual, (x, y), upstream)
    expected_grads = torch.autograd.grad(expected, (x_ref, y_ref), upstream)
    # The existing wrapper stages accumulated costs through float32, even
    # for float64 inputs; use its established numerical precision here.
    rtol, atol = 2e-5, 2e-6
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            actual_grad, expected_grad, rtol=rtol, atol=atol
        )


def test_soft_dtw_keeps_shared_distance_matrix_unchanged():
    """Path masking preserves costs reused by another objective."""
    x = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]], requires_grad=True)
    y = torch.tensor([[[0.5], [1.5], [2.5], [3.5]]], requires_grad=True)
    distance = torch.cdist(x, y)
    saved = distance.detach().clone()
    loss = _SoftDTWLossPyTorch.apply(distance, 1.0, 2, 1, None)
    torch.testing.assert_close(distance, saved, rtol=0, atol=0)
    (loss.sum() + distance.square().mean()).backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(y.grad).all()
