"""Small CPU smoke test for forward, backward, and checkpoint round-trip."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from radiant_config import RADIANTConfig
from radiant_model import RADIANT, RADIANTLoss


def build_model():
    return RADIANT(
        RADIANTConfig.s,
        RADIANTConfig.n,
        RADIANTConfig.channel,
        RADIANTConfig.stride,
        num_tasks=RADIANTConfig.num_tasks,
    )


def main():
    torch.manual_seed(7)
    model = build_model().cpu().train()
    criterion = RADIANTLoss()
    source_1, source_2, near, far = [
        torch.rand(2, 1, 16, 16) for _ in range(4)
    ]

    outputs = model.forward_with_radiant(
        source_1,
        source_2,
        x_mfif=near,
        y_mfif=far,
        training_tag=1,
        task_id=0,
    )
    loss, pieces = criterion(
        outputs,
        {"source1": source_1, "source2": source_2, "target_rec": source_2},
    )
    loss.backward()

    if outputs["out_f"].shape != source_1.shape:
        raise AssertionError("fused output shape does not match the input")
    if not torch.isfinite(loss):
        raise AssertionError("loss is not finite")
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("missing or non-finite gradients")

    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "smoke.pt"
        torch.save({"model_state": model.state_dict()}, checkpoint)
        restored = build_model().cpu().eval()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restored.load_state_dict(payload["model_state"])
        with torch.inference_mode():
            fused = restored(source_1[:1], source_2[:1], task_id=0, update_memory=False)
        if fused.shape != source_1[:1].shape or not torch.isfinite(fused).all():
            raise AssertionError("checkpoint round-trip inference failed")

    summary = ", ".join(f"{name}={float(value.detach()):.4f}" for name, value in pieces.items())
    print(f"SMOKE TEST PASSED | loss={float(loss.detach()):.4f} | {summary}")


if __name__ == "__main__":
    main()
