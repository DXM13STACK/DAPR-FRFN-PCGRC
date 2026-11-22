import torch
import torch.nn as nn

from mmseg.registry import MODELS

from projects.industrial_defect_seg.models.dapr_frfn_pcgrc_segformer_head import (
    DAPRFRFNPCGRCSegformerHead,
)


class _UnitChannelGate(nn.Module):
    """Remove channel gating while keeping the parent's forward unchanged.

    The gated implementation produces a channel-wise gate and multiplies
    it with the PCGRC residual correction. Returning all ones therefore gives

        gate * delta_pcgrc = delta_pcgrc

    which is exactly the naive residual addition required by this ablation.
    """

    def forward(self, x):
        if x.ndim != 4:
            raise RuntimeError(
                'Unit PCGRC gate expects BCHW input, '
                f'got shape {tuple(x.shape)}.'
            )

        return x.new_ones(
            (
                x.shape[0],
                x.shape[1],
                1,
                1,
            )
        )


@MODELS.register_module()
class DAPRFRFNNaiveCorrectionSegformerHead(
    DAPRFRFNPCGRCSegformerHead
):
    """Strict DAPR-FRFN + naive correction ablation.

    Everything is inherited from the verified PCGRC implementation:

        MiT-B0
          -> SegFormer projection/fusion
          -> DAPR
          -> FRFN refinement
          -> PCGRC coarse classifier
          -> PCGRCResidualCorrection

    The ONLY ablation is:

        channel gate  -> constant 1
        pcgrc_gamma     -> fixed 1

    Therefore:

        F_out
          = F_DAPR
            + 1 * (1 * Delta_F_PCGRC)

          = F_DAPR + Delta_F_PCGRC

    No new convolution, attention, loss, auxiliary head, or routing
    mechanism is introduced here.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


        self.pcgrc_channel_gate = _UnitChannelGate()


        if not hasattr(self, 'pcgrc_gamma'):
            raise RuntimeError(
                'Parent PCGRC head has no attribute "pcgrc_gamma". '
                'The current server implementation does not match the '
                'expected verified PCGRC version.'
            )

        if not torch.is_tensor(self.pcgrc_gamma):
            raise RuntimeError(
                'Expected pcgrc_gamma to be a torch Tensor/Parameter, '
                f'got {type(self.pcgrc_gamma)}.'
            )

        with torch.no_grad():
            self.pcgrc_gamma.fill_(1.0)

        self.pcgrc_gamma.requires_grad_(False)
