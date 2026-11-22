import torch.nn as nn

from mmseg.registry import MODELS

from .dapr_frfn_pcgrc_segformer_head import (
    DAPRFRFNPCGRCSegformerHead,
)


class _UnitChannelGate(nn.Module):
    """Return ones without learnable channel gating."""

    def forward(self, x):
        return x.new_ones(
            (x.shape[0], x.shape[1], 1, 1)
        )


@MODELS.register_module()
class DAPRFRFNScaleOnlyCorrectionSegformerHead(
    DAPRFRFNPCGRCSegformerHead
):
    """PCGRC residual correction with learnable scalar scaling only.

    Inherited PCGRC:
        F_out = F + gamma * gate(F) * delta

    Scale-only:
        F_out = F + gamma * delta

    The PCGRC branch and learnable gamma are preserved.
    Only the channel gate is replaced by constant ones.
    """

    def __init__(self, pcgrc_gamma_init=0.01, **kwargs):
        super().__init__(
            pcgrc_gamma_init=pcgrc_gamma_init,
            **kwargs,
        )

        self.pcgrc_channel_gate = _UnitChannelGate()
