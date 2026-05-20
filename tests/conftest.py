from __future__ import annotations

import os

# Force-disable W&B for all pytest tests, including subprocess-based CLI tests.
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
