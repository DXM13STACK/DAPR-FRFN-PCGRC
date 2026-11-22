import torch
import torch.nn as nn

from mmseg.models.decode_heads.segformer_head import SegformerHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS


class FRFN(nn.Module):

    def __init__(
        self,
        dim,
        hidden_dim,
        act_layer=nn.GELU,
    ):
        super().__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim


        self.linear1 = nn.Sequential(
            nn.Linear(
                dim,
                hidden_dim * 2,
            ),
            act_layer(),
        )


        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=hidden_dim,
            ),
            act_layer(),
        )


        self.linear2 = nn.Linear(
            hidden_dim,
            dim,
        )


        self.dim_conv = self.dim // 4
        self.dim_untouched = (
            self.dim - self.dim_conv
        )

        assert self.dim_conv > 0

        self.partial_conv3 = nn.Conv2d(
            self.dim_conv,
            self.dim_conv,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(
        self,
        x,
        h,
        w,
    ):
        """Forward.

        Args
        ----
        x:
            [B, H*W, C]

        h, w:
            real spatial feature size.

        Returns
        -------
        x:
            [B, H*W, C]
        """

        b, hw, c = x.shape

        assert c == self.dim, (
            f'FRFN channel mismatch: '
            f'expected {self.dim}, got {c}.'
        )

        assert hw == h * w, (
            f'FRFN spatial mismatch: '
            f'H*W={h*w}, token length={hw}.'
        )


        x = x.transpose(
            1,
            2,
        ).reshape(
            b,
            c,
            h,
            w,
        )


        x1, x2 = torch.split(
            x,
            [
                self.dim_conv,
                self.dim_untouched,
            ],
            dim=1,
        )

        x1 = self.partial_conv3(x1)

        x = torch.cat(
            (x1, x2),
            dim=1,
        )


        x = x.flatten(2).transpose(
            1,
            2,
        ).contiguous()


        x = self.linear1(x)


        x_1, x_2 = x.chunk(
            2,
            dim=-1,
        )


        x_1 = x_1.transpose(
            1,
            2,
        ).reshape(
            b,
            self.hidden_dim,
            h,
            w,
        )

        x_1 = self.dwconv(x_1)

        x_1 = x_1.flatten(2).transpose(
            1,
            2,
        ).contiguous()


        x = x_1 * x_2


        x = self.linear2(x)

        return x


class FRFNRefinementBlock(nn.Module):

    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
    ):
        super().__init__()

        hidden_dim = int(
            dim * mlp_ratio
        )

        self.norm = nn.LayerNorm(dim)

        self.frfn = FRFN(
            dim=dim,
            hidden_dim=hidden_dim,
            act_layer=nn.GELU,
        )

    def forward(self, x):
        """Args:
            x: [B, C, H, W]

        Returns:
            [B, C, H, W]
        """

        b, c, h, w = x.shape


        tokens = x.flatten(2).transpose(
            1,
            2,
        ).contiguous()


        shortcut = tokens

        tokens = self.norm(tokens)

        tokens = self.frfn(
            tokens,
            h=h,
            w=w,
        )

        tokens = shortcut + tokens


        x = tokens.transpose(
            1,
            2,
        ).reshape(
            b,
            c,
            h,
            w,
        )

        return x


@MODELS.register_module()
class FRFNSegformerHead(SegformerHead):
    """SegFormer decode head with original FRFN refinement.

    Baseline SegFormer:

        F1/F2/F3/F4
             ↓
        linear projections
             ↓
          resize
             ↓
           concat
             ↓
        fusion_conv
             ↓
           cls_seg

    Current screening model:

        F1/F2/F3/F4
             ↓
        original SegFormer
        projections/resize
             ↓
           concat
             ↓
        fusion_conv
             ↓
           FRFN
             ↓
           cls_seg

    Therefore:

    - backbone unchanged
    - SegFormer multi-scale fusion unchanged
    - FRFN core unchanged
    - loss unchanged
    - classifier unchanged

    Only one original FRFN refinement block is inserted
    after SegFormer's fusion_conv.
    """

    def __init__(
        self,
        frfn_mlp_ratio=4.0,
        **kwargs,
    ):
        super().__init__(**kwargs)


        dim = self.channels

        assert dim % 4 == 0, (
            'FRFN requires decoder channels divisible by 4 '
            'because the original partial convolution uses '
            '1/4 of the channels. '
            f'Current channels={dim}.'
        )

        self.frfn_refine = FRFNRefinementBlock(
            dim=dim,
            mlp_ratio=frfn_mlp_ratio,
        )

    def forward(self, inputs):


        inputs = self._transform_inputs(inputs)

        assert len(inputs) == 4

        target_size = inputs[0].shape[2:]

        outs = []


        for idx in range(len(inputs)):

            x = inputs[idx]

            conv = self.convs[idx]

            x = conv(x)


            x = resize(
                input=x,
                size=target_size,
                mode=self.interpolate_mode,
                align_corners=self.align_corners,
            )

            outs.append(x)


        out = torch.cat(
            outs,
            dim=1,
        )


        out = self.fusion_conv(out)


        out = self.frfn_refine(out)


        out = self.cls_seg(out)

        return out
