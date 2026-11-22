import torch
import torch.nn as nn

from mmseg.registry import MODELS
from mmseg.models.utils import resize

from .frfn_segformer_head import FRFNSegformerHead


class StatisticalChannelRecalibration(nn.Module):

    def __init__(
        self,
        channels,
        reduction=16,
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            1
        )

        self.channels = channels

        self.mlp = nn.Sequential(
            nn.Linear(
                channels * 2,
                hidden
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Linear(
                hidden,
                channels
            ),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x:
            [B, C, H, W]

        return:
            recalibrated feature [B,C,H,W]
            weight [B,C,1,1]
        """


        mean = torch.mean(
            x,
            dim=(2, 3)
        )


        std = torch.std(
            x,
            dim=(2, 3),
            unbiased=False
        )


        descriptor = torch.cat(
            [mean, std],
            dim=1
        )


        weight = self.mlp(
            descriptor
        )

        weight = self.sigmoid(
            weight
        )


        weight = weight.unsqueeze(-1).unsqueeze(-1)

        out = x * weight

        return out, weight


class _StatsAblationFRFNBase(
    FRFNSegformerHead
):

    def _fusion_feature(
        self,
        inputs
    ):

        inputs = self._transform_inputs(
            inputs
        )

        outs = []

        for idx in range(
            len(inputs)
        ):

            x = inputs[idx]

            conv = self.convs[idx]

            x = conv(x)

            x = resize(
                input=x,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners,
            )

            outs.append(x)

        out = torch.cat(
            outs,
            dim=1
        )

        out = self.fusion_conv(
            out
        )

        return out


@MODELS.register_module()
class StatsCAFRFNSegformerHead(
    _StatsAblationFRFNBase
):

    def __init__(
        self,
        stats_reduction=16,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )

        self.stats_reduction = (
            stats_reduction
        )

        self.stats_ca = (
            StatisticalChannelRecalibration(
                channels=self.channels,
                reduction=stats_reduction,
            )
        )

    def forward(
        self,
        inputs
    ):


        out = self._fusion_feature(
            inputs
        )


        out, _ = self.stats_ca(
            out
        )


        out = self.frfn_refine(
            out
        )


        out = self.cls_seg(
            out
        )

        return out


@MODELS.register_module()
class StatsCAPConvFRFNSegformerHead(
    _StatsAblationFRFNBase
):

    def __init__(
        self,
        stats_reduction=16,
        partial_div=4,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )

        assert (
            self.channels
            % partial_div
            == 0
        )

        self.stats_reduction = (
            stats_reduction
        )

        self.partial_div = (
            partial_div
        )

        self.partial_channels = (
            self.channels
            // partial_div
        )

        self.stats_ca = (
            StatisticalChannelRecalibration(
                channels=self.channels,
                reduction=stats_reduction,
            )
        )


        self.partial_conv = nn.Conv2d(
            self.partial_channels,
            self.partial_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(
        self,
        inputs
    ):


        out = self._fusion_feature(
            inputs
        )


        out, _ = self.stats_ca(
            out
        )


        x_partial = out[
            :,
            :self.partial_channels,
            :,
            :
        ]

        x_bypass = out[
            :,
            self.partial_channels:,
            :,
            :
        ]

        x_partial = self.partial_conv(
            x_partial
        )

        out = torch.cat(
            [
                x_partial,
                x_bypass
            ],
            dim=1
        )


        out = self.frfn_refine(
            out
        )


        out = self.cls_seg(
            out
        )

        return out
