"""Core RADIANT information-allocation modules.

These modules provide the trainable allocation path used by ``RADIANT``.
Keeping the production path in a focused module makes the implementation easier
to inspect and gives training and inference the same well-defined interfaces.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _orthonormal_columns(matrix: Tensor) -> Tensor:
    """Return a differentiable semi-orthogonal basis for a tall matrix."""

    q, r = torch.linalg.qr(matrix, mode="reduced")
    # QR has a sign ambiguity. Fixing the sign makes checkpointed inference
    # deterministic while retaining gradients almost everywhere.
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0)


class NonInterferingDecomposition(nn.Module):
    """Matrix-exponential shared/unique feature decomposition.

    The skew-symmetric generators produce stable rotations. Sequential
    complement projections discourage the shared and two unique streams from
    repeatedly encoding the same evidence.
    """

    def __init__(self, input_dim: int, stream_dim: int = 32) -> None:
        super().__init__()
        if stream_dim * 3 > input_dim:
            raise ValueError("stream_dim * 3 must not exceed input_dim")

        self.input_dim = input_dim
        self.stream_dim = stream_dim
        self.generators = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim, input_dim)) for _ in range(3)]
        )
        self.bases = nn.ParameterList(
            [nn.Parameter(torch.empty(input_dim, stream_dim)) for _ in range(3)]
        )
        for basis in self.bases:
            nn.init.orthogonal_(basis)

    def _rotated(self, index: int, complement: Tensor) -> Tensor:
        generator = self.generators[index]
        rotation = torch.matrix_exp(generator - generator.transpose(0, 1))
        return _orthonormal_columns(rotation @ complement @ self.bases[index])

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        batch, channels, height, width = features.shape
        if channels != self.input_dim:
            raise ValueError(f"expected {self.input_dim} channels, got {channels}")

        identity = torch.eye(self.input_dim, device=features.device, dtype=features.dtype)
        q_shared = self._rotated(0, identity)
        complement_1 = identity - q_shared @ q_shared.transpose(0, 1)
        q_unique_1 = self._rotated(1, complement_1)
        complement_2 = (
            complement_1 - q_unique_1 @ q_unique_1.transpose(0, 1)
        )
        q_unique_2 = self._rotated(2, complement_2)

        tokens = features.flatten(2).transpose(1, 2)

        def project(basis: Tensor) -> Tensor:
            stream = tokens @ basis
            return stream.transpose(1, 2).reshape(
                batch, self.stream_dim, height, width
            )

        shared = project(q_shared)
        unique_1 = project(q_unique_1)
        unique_2 = project(q_unique_2)

        bases = (q_shared, q_unique_1, q_unique_2)
        overlap = features.new_zeros(())
        for index, left in enumerate(bases):
            for right in bases[index + 1 :]:
                overlap = overlap + (left.transpose(0, 1) @ right).square().mean()

        stream_identity = torch.eye(
            self.stream_dim, device=features.device, dtype=features.dtype
        )
        semi_orthogonality = sum(
            ((basis.transpose(0, 1) @ basis) - stream_identity).square().mean()
            for basis in bases
        )
        losses = {
            "overlap": overlap,
            "semi_orthogonality": semi_orthogonality,
        }
        return shared, unique_1, unique_2, losses


def _contrastive_loss(left: Tensor, right: Tensor, max_tokens: int = 256) -> Tensor:
    """Symmetric token-level InfoNCE with bounded memory use."""

    left_tokens = left.flatten(2).transpose(1, 2).reshape(-1, left.shape[1])
    right_tokens = right.flatten(2).transpose(1, 2).reshape(-1, right.shape[1])
    count = left_tokens.shape[0]
    if count > max_tokens:
        indices = torch.linspace(
            0, count - 1, max_tokens, device=left.device
        ).round().long()
        left_tokens = left_tokens.index_select(0, indices)
        right_tokens = right_tokens.index_select(0, indices)

    left_tokens = F.normalize(left_tokens, dim=1)
    right_tokens = F.normalize(right_tokens, dim=1)
    if left_tokens.shape[0] == 1:
        return 1.0 - (left_tokens * right_tokens).sum(dim=1).mean()

    logits = left_tokens @ right_tokens.transpose(0, 1) / 0.07
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )


class MultiGranularityInteraction(nn.Module):
    """Compose pixel-, block-, and global-scale modality interactions."""

    def __init__(
        self,
        channels: int,
        block_size: int = 4,
        ema_momentum: float = 0.95,
        weight_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.ema_momentum = ema_momentum
        self.weight_temperature = weight_temperature
        self.left_projection = nn.Conv2d(channels, channels, 1)
        self.right_projection = nn.Conv2d(channels, channels, 1)
        self.output_projection = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )
        self.register_buffer("ema_scores", torch.zeros(3))
        self.register_buffer("ema_initialized", torch.tensor(False))

    def forward(self, unique_1: Tensor, unique_2: Tensor) -> Dict[str, Tensor]:
        left = self.left_projection(unique_1)
        right = self.right_projection(unique_2)
        height, width = left.shape[-2:]

        pixel = torch.tanh(left) * torch.sigmoid(right) + torch.tanh(right) * torch.sigmoid(left)

        pool_h = min(self.block_size, height)
        pool_w = min(self.block_size, width)
        left_block = F.adaptive_avg_pool2d(left, (pool_h, pool_w))
        right_block = F.adaptive_avg_pool2d(right, (pool_h, pool_w))
        block_small = torch.tanh(left_block) * torch.sigmoid(right_block)
        block = F.interpolate(block_small, (height, width), mode="bilinear", align_corners=False)

        left_global = F.adaptive_avg_pool2d(left, 1)
        right_global = F.adaptive_avg_pool2d(right, 1)
        global_map = (torch.tanh(left_global) * torch.sigmoid(right_global)).expand_as(left)

        pixel_loss = _contrastive_loss(left, right)
        block_loss = _contrastive_loss(left_block, right_block)
        global_loss = 1.0 - F.cosine_similarity(
            left_global.flatten(1), right_global.flatten(1), dim=1
        ).mean()
        spectral_regularizer = F.l1_loss(left_global, right_global)
        global_loss = global_loss + 0.1 * spectral_regularizer
        losses = torch.stack((pixel_loss, block_loss, global_loss))

        # The paper uses EMA-smoothed MI ranking surrogates. Here the bounded
        # InfoNCE objectives provide that surrogate without a separate density
        # estimator; lower loss means a stronger interaction scale.
        # Pixel, block, and global objectives have different candidate counts,
        # so compare a bounded log-loss surrogate instead of raw magnitudes.
        scores = -0.05 * torch.log1p(losses.detach().clamp_min(0.0))
        if self.training:
            with torch.no_grad():
                if not bool(self.ema_initialized.item()):
                    self.ema_scores.copy_(scores)
                    self.ema_initialized.fill_(True)
                else:
                    self.ema_scores.mul_(self.ema_momentum).add_(
                        scores, alpha=1.0 - self.ema_momentum
                    )

        ranking_scores = self.ema_scores if bool(self.ema_initialized.item()) else scores
        weights = F.softmax(ranking_scores / self.weight_temperature, dim=0)
        synergy = (
            weights[0] * pixel + weights[1] * block + weights[2] * global_map
        )
        synergy = self.output_projection(synergy)
        interaction_loss = (weights.detach() * losses).sum()

        return {
            "synergy": synergy,
            "interaction_loss": interaction_loss,
            "pixel_loss": pixel_loss,
            "block_loss": block_loss,
            "global_loss": global_loss,
            "scale_weights": weights,
        }


class TaskMemory(nn.Module):
    """Momentum task memory with hard-positive updates and repulsion."""

    def __init__(
        self,
        num_tasks: int,
        channels: int,
        momentum: float = 0.97,
        noise_std: float = 0.14,
        temperature: float = 0.07,
        hard_threshold_degrees: float = 60.0,
    ) -> None:
        super().__init__()
        if num_tasks < 1:
            raise ValueError("num_tasks must be positive")
        self.num_tasks = num_tasks
        self.channels = channels
        self.momentum = momentum
        self.noise_std = noise_std
        self.temperature = temperature
        self.hard_threshold = math.cos(math.radians(hard_threshold_degrees))

        memory = F.normalize(torch.randn(num_tasks, channels), dim=1)
        self.register_buffer("memory", memory)
        self.query_projection = nn.Linear(channels, channels)
        self.context_projection = nn.Linear(channels, channels)

    def _task_ids(self, task_id: int | Tensor, batch: int, device: torch.device) -> Tensor:
        if isinstance(task_id, int):
            ids = torch.full((batch,), task_id, dtype=torch.long, device=device)
        else:
            ids = task_id.to(device=device, dtype=torch.long).reshape(-1)
            if ids.numel() == 1:
                ids = ids.expand(batch)
        if ids.numel() != batch:
            raise ValueError("task_id must be a scalar or contain one id per sample")
        if ids.min() < 0 or ids.max() >= self.num_tasks:
            raise ValueError(f"task ids must be in [0, {self.num_tasks - 1}]")
        return ids

    def forward(
        self,
        synergy: Tensor,
        task_id: int | Tensor,
        update_memory: bool = True,
    ) -> Dict[str, Tensor]:
        batch = synergy.shape[0]
        task_ids = self._task_ids(task_id, batch, synergy.device)
        queries = F.normalize(
            self.query_projection(F.adaptive_avg_pool2d(synergy, 1).flatten(1)),
            dim=1,
        )
        # Clone because the momentum update below mutates the registered buffer
        # before backward; autograd still needs the values for query gradients.
        normalized_memory = F.normalize(self.memory.detach().clone(), dim=1)
        logits = queries @ normalized_memory.transpose(0, 1) / self.temperature
        alignment_loss = F.cross_entropy(logits, task_ids)

        positive_similarity = (queries * normalized_memory[task_ids]).sum(dim=1)
        hard_mask = positive_similarity < self.hard_threshold

        if self.training and update_memory:
            with torch.no_grad():
                for current_task in task_ids.unique():
                    task_mask = task_ids == current_task
                    selected = task_mask & hard_mask
                    if not selected.any():
                        selected = task_mask
                    update = F.normalize(queries[selected].mean(dim=0), dim=0)
                    noise = torch.randn_like(update) * self.noise_std
                    evolved = (
                        self.momentum * self.memory[current_task]
                        + (1.0 - self.momentum) * update
                        + noise
                    )
                    self.memory[current_task].copy_(F.normalize(evolved, dim=0))

        attention = F.softmax(logits, dim=1)
        context = attention @ normalized_memory
        gate = torch.sigmoid(self.context_projection(context)).view(
            batch, self.channels, 1, 1
        )
        conditioned = synergy * (1.0 + gate)

        similarity = normalized_memory @ normalized_memory.transpose(0, 1)
        off_diagonal = ~torch.eye(
            self.num_tasks, dtype=torch.bool, device=synergy.device
        )
        if off_diagonal.any():
            repulsion_loss = F.relu(similarity[off_diagonal] - 0.2).square().mean()
        else:
            repulsion_loss = synergy.new_zeros(())

        return {
            "conditioned_synergy": conditioned,
            "alignment_loss": alignment_loss,
            "repulsion_loss": repulsion_loss,
            "hard_mask": hard_mask,
            "queries": queries,
            "logits": logits,
            "attention": attention,
        }


class SobelGradient(nn.Module):
    """Differentiable per-channel Sobel gradient magnitude."""

    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ) / 4.0
        kernel_y = kernel_x.transpose(0, 1)
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3))

    def forward(self, image: Tensor) -> Tensor:
        channels = image.shape[1]
        kernel_x = self.kernel_x.expand(channels, 1, 3, 3)
        kernel_y = self.kernel_y.expand(channels, 1, 3, 3)
        grad_x = F.conv2d(image, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(image, kernel_y, padding=1, groups=channels)
        return grad_x.abs() + grad_y.abs()
