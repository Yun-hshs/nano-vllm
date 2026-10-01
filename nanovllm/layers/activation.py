import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers import fused_ops


class SiluAndMul(nn.Module):

    @torch.compile
    def _compiled_forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if fused_ops.fused_enabled():
            # Single-pass Triton: reads the gate and up halves of the row and
            # writes only the product, instead of materialising silu(gate).
            return fused_ops.silu_and_mul(x)
        return self._compiled_forward(x)
