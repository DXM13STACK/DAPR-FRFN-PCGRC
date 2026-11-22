import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.decode_heads.segformer_head import SegformerHead
from mmseg.models.utils import resize


class PCGRCResidualCorrection(nn.Module):

    def __init__(
        self,
        channels,
        eps=1e-6,
    ):
        super().__init__()

        self.channels = channels
        self.eps = eps

        self.psi = nn.Sequential(
            nn.Conv2d(
                channels + 2,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def _foreground_probability(self, coarse_logits):

        if coarse_logits.ndim != 4:
            raise RuntimeError(
                'PCGRC expects BCHW coarse logits, '
                f'got shape {tuple(coarse_logits.shape)}.'
            )


        if coarse_logits.shape[1] == 1:
            return torch.sigmoid(coarse_logits)


        if coarse_logits.shape[1] == 2:
            return torch.softmax(
                coarse_logits,
                dim=1,
            )[:, 1:2]

        raise RuntimeError(
            'Current PCGRC reconstruction is for binary '
            'segmentation. Expected 1 or 2 output channels, '
            f'got {coarse_logits.shape[1]}.'
        )

    def _confidence(self, probability):

        p = probability.clamp(
            min=self.eps,
            max=1.0 - self.eps,
        )

        entropy = -(
            p * torch.log(p)
            +
            (1.0 - p) * torch.log(1.0 - p)
        )

        confidence = (
            1.0
            -
            entropy / math.log(2.0)
        )

        return confidence.clamp(
            min=0.0,
            max=1.0,
        )

    @staticmethod
    def _boundary_response(probability):


        dx = torch.abs(
            probability[:, :, :, 1:]
            -
            probability[:, :, :, :-1]
        )

        dx = F.pad(
            dx,
            pad=(0, 1, 0, 0),
            mode='constant',
            value=0.0,
        )


        dy = torch.abs(
            probability[:, :, 1:, :]
            -
            probability[:, :, :-1, :]
        )

        dy = F.pad(
            dy,
            pad=(0, 0, 0, 1),
            mode='constant',
            value=0.0,
        )

        return dx + dy

    def forward(
        self,
        feature,
        coarse_logits,
    ):

        if feature.ndim != 4:
            raise RuntimeError(
                'PCGRC expects BCHW feature map, '
                f'got shape {tuple(feature.shape)}.'
            )

        if feature.shape[1] != self.channels:
            raise RuntimeError(
                f'PCGRC expected {self.channels} feature '
                f'channels, got {feature.shape[1]}.'
            )


        probability = self._foreground_probability(
            coarse_logits
        )

        if probability.shape[2:] != feature.shape[2:]:
            probability = F.interpolate(
                probability,
                size=feature.shape[2:],
                mode='bilinear',
                align_corners=False,
            )


        confidence = self._confidence(
            probability
        )

        boundary = self._boundary_response(
            probability
        )


        correction_input = torch.cat(
            [
                feature,
                confidence,
                boundary,
            ],
            dim=1,
        )

        correction = self.psi(
            correction_input
        )

        return feature + correction


@MODELS.register_module()
class PCGRCReproMTDSegformerHead(SegformerHead):

    def __init__(
        self,
        pcgrc_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)


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

    def _project_and_align(
        self,
        inputs,
    ):

        target_size = inputs[0].shape[2:]

        outs = []

        for idx, x in enumerate(inputs):


            x = self.convs[idx](x)


            x = resize(
                input=x,
                size=target_size,
                mode=self.interpolate_mode,
                align_corners=self.align_corners,
            )

            outs.append(x)


        fused = torch.cat(
            outs,
            dim=1,
        )


        fused = self.fusion_conv(
            fused
        )

        return fused

    def forward(
        self,
        inputs,
    ):

        inputs = self._transform_inputs(
            inputs
        )

        if len(inputs) != 4:
            raise RuntimeError(
                'PCGRC reproduction expects exactly four '
                f'MiT feature levels, got {len(inputs)}.'
            )


        fused = self._project_and_align(
            inputs
        )


        coarse_logits = self.pcgrc_coarse_classifier(
            fused
        )


        refined = self.pcgrc(
            fused,
            coarse_logits,
        )


        return self.cls_seg(
            refined
        )
