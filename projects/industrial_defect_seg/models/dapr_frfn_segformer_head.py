import torch
import torch.nn as nn

from mmseg.registry import MODELS
from mmseg.models.decode_heads.segformer_head import SegformerHead
from mmseg.models.utils import resize


class DAPRFRFN(nn.Module):

    def __init__(
        self,
        dim=256,
        hidden_dim=1024,
        reduction=16,
        partial_div=4,
        gamma_init=0.01,
        act_layer=nn.GELU,
    ):
        super().__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim


        self.dim_conv = dim // partial_div
        self.dim_untouched = dim - self.dim_conv

        assert self.dim_conv > 0


        attention_hidden = max(dim // reduction, 8)

        self.channel_router = nn.Sequential(
            nn.Linear(dim * 2, attention_hidden),
            act_layer(),
            nn.Linear(attention_hidden, dim),
            nn.Sigmoid()
        )


        self.route_proj = nn.Conv2d(
            dim,
            self.dim_conv,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )


        self.gamma = nn.Parameter(
            torch.tensor(float(gamma_init))
        )


        self.partial_conv3 = nn.Conv2d(
            self.dim_conv,
            self.dim_conv,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )


        self.linear1 = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            act_layer()
        )


        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=hidden_dim
            ),
            act_layer()
        )


        self.linear2 = nn.Linear(
            hidden_dim,
            dim
        )

    def forward(self, x, h, w):
        """
        Args:
            x: [B, N, C]
               N = H * W

            h, w:
               显式传入空间尺寸。
               这样支持 KSDD2 的 64x160，
               不使用原 AST 的 sqrt(N) 假设。

        Returns:
            x: [B, N, C]
        """

        b, n, c = x.shape

        assert c == self.dim,\
            f'Expected channel={self.dim}, got {c}'

        assert n == h * w,\
            f'N={n}, but H*W={h*w} ({h}x{w})'


        x_map = (
            x.transpose(1, 2)
             .reshape(b, c, h, w)
             .contiguous()
        )


        mean = x_map.mean(
            dim=(2, 3)
        )


        centered = (
            x_map
            - mean.unsqueeze(-1).unsqueeze(-1)
        )

        var = (
            centered.pow(2)
                    .mean(dim=(2, 3))
        )

        std = torch.sqrt(
            var + 1e-6
        )


        descriptor = torch.cat(
            [mean, std],
            dim=1
        )


        channel_weight = self.channel_router(
            descriptor
        )

        channel_weight = channel_weight.view(
            b,
            c,
            1,
            1
        )


        routed = x_map * channel_weight

        routed = self.route_proj(
            routed
        )


        x_partial, x_untouched = torch.split(
            x_map,
            [
                self.dim_conv,
                self.dim_untouched
            ],
            dim=1
        )


        x_partial = (
            x_partial
            + self.gamma * routed
        )


        x_partial = self.partial_conv3(
            x_partial
        )


        x_map = torch.cat(
            [
                x_partial,
                x_untouched
            ],
            dim=1
        )


        x = (
            x_map.flatten(2)
                 .transpose(1, 2)
                 .contiguous()
        )


        x = self.linear1(x)


        x1, x2 = x.chunk(
            2,
            dim=-1
        )


        x1 = (
            x1.transpose(1, 2)
              .reshape(
                  b,
                  self.hidden_dim,
                  h,
                  w
              )
              .contiguous()
        )

        x1 = self.dwconv(x1)

        x1 = (
            x1.flatten(2)
              .transpose(1, 2)
              .contiguous()
        )


        x = x1 * x2


        x = self.linear2(x)

        return x


class DAPRFRFNRefinementBlock(nn.Module):

    def __init__(
        self,
        dim=256,
        mlp_ratio=4.0,
        reduction=16,
        partial_div=4,
        gamma_init=0.01,
    ):
        super().__init__()

        hidden_dim = int(
            dim * mlp_ratio
        )

        self.norm = nn.LayerNorm(dim)

        self.frfn = DAPRFRFN(
            dim=dim,
            hidden_dim=hidden_dim,
            reduction=reduction,
            partial_div=partial_div,
            gamma_init=gamma_init,
        )

    def forward(self, x):
        """
        x:
            [B,C,H,W]
        """

        b, c, h, w = x.shape

        tokens = (
            x.flatten(2)
             .transpose(1, 2)
             .contiguous()
        )

        shortcut = tokens


        tokens = self.norm(tokens)


        tokens = self.frfn(
            tokens,
            h,
            w
        )


        tokens = (
            shortcut
            + tokens
        )

        out = (
            tokens.transpose(1, 2)
                  .reshape(
                      b,
                      c,
                      h,
                      w
                  )
                  .contiguous()
        )

        return out


@MODELS.register_module()
class DAPRFRFNSegformerHead(SegformerHead):

    def __init__(
        self,
        frfn_mlp_ratio=4.0,
        dapr_reduction=16,
        dapr_partial_div=4,
        dapr_gamma_init=0.01,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.frfn_refine = DAPRFRFNRefinementBlock(
            dim=self.channels,
            mlp_ratio=frfn_mlp_ratio,
            reduction=dapr_reduction,
            partial_div=dapr_partial_div,
            gamma_init=dapr_gamma_init,
        )

    def forward(self, inputs):


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
                mode=self.interpolate_mode,
                align_corners=self.align_corners
            )

            outs.append(x)


        out = torch.cat(
            outs,
            dim=1
        )

        out = self.fusion_conv(
            out
        )


        out = self.frfn_refine(
            out
        )


        out = self.cls_seg(
            out
        )

        return out
