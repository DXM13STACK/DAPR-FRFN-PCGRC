import torch
import torch.nn as nn

from mmseg.registry import MODELS

from projects.industrial_defect_seg.models.dapr_frfn_segformer_head import (
    DAPRFRFNSegformerHead,
)
from projects.industrial_defect_seg.models.pcgrc_module import (
    PCGRCResidualCorrection,
)


def _to_binary_pcgrc_logits(coarse_logits):
    """
    Convert multiclass coarse logits into binary
    background/foreground logits for PCGRC only.

    Class 0:
        background

    Classes 1..C-1:
        foreground

    For C <= 2, return the original logits unchanged.

    For C > 2:

        z_bg = z_0

        z_fg = logsumexp(
            z_1, z_2, ..., z_{C-1}
        )

    This conversion is only used by the PCGRC cue branch.
    Final semantic prediction still uses the original
    multiclass cls_seg classifier.
    """

    if coarse_logits.ndim != 4:
        raise RuntimeError(
            'Expected BCHW coarse logits before PCGRC, '
            f'got shape {tuple(coarse_logits.shape)}.'
        )

    num_channels = coarse_logits.shape[1]


    if num_channels <= 2:
        return coarse_logits

    background_logits = coarse_logits[:, 0:1]

    foreground_logits = torch.logsumexp(
        coarse_logits[:, 1:],
        dim=1,
        keepdim=True,
    )

    return torch.cat(
        [
            background_logits,
            foreground_logits,
        ],
        dim=1,
    )


@MODELS.register_module()
class DAPRFRFNPCGRCSegformerHead(DAPRFRFNSegformerHead):
    """
    DAPR-FRFN + controlled PCGRC residual injection.

    DAPR feature:
        F_d

    PCGRC:
        F_pcgrc = F_d + Delta_pcgrc

    Channel gate:
        g = sigmoid(MLP(GAP(F_d)))

    Final:
        F_out = F_d + gamma * g * Delta_pcgrc

    The original DAPR-FRFN implementation is not modified.
    The original PCGRCResidualCorrection implementation is not modified.

    For multiclass segmentation, coarse semantic logits are
    reduced to background-vs-foreground logits only for the
    PCGRC structural-cue branch.

    Final semantic prediction remains multiclass.
    """

    def __init__(
        self,
        pcgrc_eps=1e-6,
        pcgrc_gate_reduction=16,
        pcgrc_gamma_init=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not hasattr(self, 'frfn_refine'):
            raise RuntimeError(
                'DAPRFRFNPCGRCSegformerHead expects the existing '
                'DAPR head to contain self.frfn_refine.'
            )

        self.pcgrc_eps = pcgrc_eps
        self.pcgrc_gate_reduction = pcgrc_gate_reduction


        self.pcgrc_coarse_classifier = nn.Conv2d(
            self.channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )


        self.pcgrc = PCGRCResidualCorrection(
            channels=self.channels,
            eps=pcgrc_eps,
        )

        hidden_channels = max(
            self.channels // pcgrc_gate_reduction,
            8,
        )


        self.pcgrc_channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                self.channels,
                hidden_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                self.channels,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )


        self.pcgrc_gamma = nn.Parameter(
            torch.tensor(
                float(pcgrc_gamma_init)
            )
        )

    def forward(self, inputs):
        """
        Preserve the exact original DAPR forward path and
        capture the output of frfn_refine, i.e. the
        DAPR-refined fused feature.

        The classifier result produced inside super().forward()
        is discarded; only its DAPR feature is reused.
        """

        captured = {}

        def _capture_dapr_feature(
            module,
            module_inputs,
            output,
        ):
            captured['feature'] = output

        handle = self.frfn_refine.register_forward_hook(
            _capture_dapr_feature
        )

        try:

            _ = super().forward(inputs)

        finally:
            handle.remove()

        if 'feature' not in captured:
            raise RuntimeError(
                'Failed to capture DAPR feature from '
                'frfn_refine.'
            )

        feature = captured['feature']


        coarse_logits = self.pcgrc_coarse_classifier(
            feature
        )


        pcgrc_logits = _to_binary_pcgrc_logits(
            coarse_logits
        )


        pcgrc_refined = self.pcgrc(
            feature,
            pcgrc_logits,
        )

        pcgrc_correction = (
            pcgrc_refined - feature
        )


        gate = self.pcgrc_channel_gate(
            feature
        )


        refined = (
            feature
            + self.pcgrc_gamma
            * gate
            * pcgrc_correction
        )


        return self.cls_seg(
            refined
        )
