import torch
import torch.nn as nn
import torch.nn.functional as F


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class PACTActQuant(nn.Module):
    """
    PACT activation quantizer.
    signed=True uses symmetric clipping in [-alpha, alpha].
    """

    def __init__(self, bitwidth=4, init_alpha=6.0, signed=True, eps=1e-8):
        super().__init__()
        self.bitwidth = int(bitwidth)
        self.signed = bool(signed)
        self.eps = float(eps)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.disabled = self.bitwidth == -1 or self.bitwidth >= 32

    def forward(self, x):
        if self.disabled:
            return x

        alpha = torch.clamp(self.alpha.abs(), min=self.eps)
        if self.signed:
            qmax = float((1 << (self.bitwidth - 1)) - 1)
            qmin = -float(1 << (self.bitwidth - 1))
            scale = alpha / qmax
            x_clip = torch.clamp(x, -alpha, alpha)
            x_int = RoundSTE.apply(x_clip / scale).clamp(qmin, qmax)
            return x_int * scale

        qmax = float((1 << self.bitwidth) - 1)
        scale = alpha / qmax
        x_clip = torch.clamp(x, 0.0, alpha)
        x_int = RoundSTE.apply(x_clip / scale).clamp(0.0, qmax)
        return x_int * scale


class SAWBWeightQuant(nn.Module):
    """
    SAWB-like weight quantizer.
    Clip value is estimated from E|w| and sqrt(E[w^2]).
    """

    COEFF = {
        2: (3.12, -2.064),
        3: (7.509, -6.892),
        4: (12.68, -12.80),
        5: (17.74, -18.64),
    }

    def __init__(self, bitwidth=4, eps=1e-8):
        super().__init__()
        self.bitwidth = int(bitwidth)
        self.eps = float(eps)
        self.disabled = self.bitwidth == -1 or self.bitwidth >= 32

    def _estimate_clip(self, w):
        mean_abs = w.abs().mean()
        mean_sq_sqrt = torch.sqrt((w * w).mean() + self.eps)
        c1, c2 = self.COEFF.get(self.bitwidth, self.COEFF[4])
        alpha = c1 * mean_abs + c2 * mean_sq_sqrt
        return torch.clamp(alpha.abs(), min=self.eps)

    def forward(self, w):
        if self.disabled:
            return w

        alpha = self._estimate_clip(w)
        qmax = float((1 << (self.bitwidth - 1)) - 1)
        qmin = -float(1 << (self.bitwidth - 1))
        scale = alpha / qmax
        w_clip = torch.clamp(w, -alpha, alpha)
        w_int = RoundSTE.apply(w_clip / scale).clamp(qmin, qmax)
        return w_int * scale


class QConv2dPACTSAWB(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
        weight_bitwidth=4,
        activation_bitwidth=4,
        act_signed=True,
        quantize_input=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.act_quant = PACTActQuant(
            bitwidth=activation_bitwidth,
            init_alpha=6.0,
            signed=act_signed,
        )
        self.weight_quant = SAWBWeightQuant(bitwidth=weight_bitwidth)
        self.quantize_input = bool(quantize_input)

    def set_input_quantization(self, enabled: bool):
        self.quantize_input = bool(enabled)

    def forward(self, x):
        qx = self.act_quant(x) if self.quantize_input else x
        qw = self.weight_quant(self.conv.weight)
        return F.conv2d(
            qx,
            qw,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.conv.groups,
        )

    def forward_shared_channel(self, x, active_in_channels, active_out_channels):
        active_in_channels = int(active_in_channels)
        active_out_channels = int(active_out_channels)

        if self.conv.groups == 1:
            x_use = x[:, :active_in_channels]
            w_use = self.conv.weight[:active_out_channels, :active_in_channels]
            b_use = self.conv.bias[:active_out_channels] if self.conv.bias is not None else None
            groups = 1
        elif self.conv.groups == self.conv.in_channels and self.conv.in_channels == self.conv.out_channels:
            active_c = min(active_in_channels, active_out_channels)
            x_use = x[:, :active_c]
            w_use = self.conv.weight[:active_c]
            b_use = self.conv.bias[:active_c] if self.conv.bias is not None else None
            groups = active_c
        else:
            raise NotImplementedError("Only groups=1 or depthwise are supported")

        qx = self.act_quant(x_use) if self.quantize_input else x_use
        qw = self.weight_quant(w_use)
        return F.conv2d(
            qx,
            qw,
            b_use,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=groups,
        )
