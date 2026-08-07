"""Default RADIANT configuration shared by model and command-line entry points.

New runs should prefer the flags exposed by ``train.py`` and ``infer.py``.
"""


class RADIANTConfig:
    # Paper-aligned training defaults (Section 4.1).
    lr = 2e-4
    epochs = 50
    batch_size = 16
    weight_decay = 5e-2
    grad_clip = 1.0
    cuda = True
    device = 0

    # Compact network configuration used by the released implementation.
    Height = 256
    Width = 256
    n = 64
    channel = 1
    s = 3
    stride = 1
    num_block = 4
    num_tasks = 6

    resume_model = None
    save_fusion_model = "./checkpoints"
    save_loss_dir = "./runs"
