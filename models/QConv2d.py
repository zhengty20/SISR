import torch
import torch.nn as nn
import torch.nn.functional as F


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        return torch.round(input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, scale):
        ctx.scale = scale
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class LSQQuantizer(nn.Module):
    def __init__(self, bitwidth, is_activation=False, per_channel=False, channels=None, eps=1e-8):
        super().__init__()
        if bitwidth < 2:
            raise ValueError("LSQ bitwidth must be >= 2")

        self.bitwidth = bitwidth
        self.is_activation = is_activation
        self.per_channel = per_channel
        self.q_n = -2 ** (bitwidth - 1)
        self.q_p = 2 ** (bitwidth - 1) - 1
        self.eps = eps
        if self.per_channel:
            if channels is None:
                raise ValueError("per_channel=True 时必须提供 channels")
            self.s = nn.Parameter(torch.ones(channels, 1, 1, 1))
        else:
            self.s = nn.Parameter(torch.tensor(1.0))
        self.initialized = False

    def _init_from_tensor(self, x):
        if self.per_channel:
            reduce_dims = tuple(range(1, x.dim()))
            mean_abs = x.detach().abs().mean(dim=reduce_dims, keepdim=True)
            s_init = (2.0 * mean_abs) / (self.q_p ** 0.5)
        else:
            mean_abs = x.detach().abs().mean()
            s_init = (2.0 * mean_abs) / (self.q_p ** 0.5)
        s_init = torch.clamp(s_init, min=self.eps)
        if self.per_channel:
            self.s.data.copy_(s_init.view_as(self.s))
        else:
            self.s.data.copy_(s_init)
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self._init_from_tensor(x)

        if self.per_channel:
            n = x[0].numel()
        else:
            n = x.numel()
        grad_scale = 1.0 / ((n * self.q_p) ** 0.5)
        s_scaled = ScaleGradient.apply(self.s, grad_scale)
        s_safe = torch.clamp(s_scaled, min=self.eps)

        x_int = x / s_safe
        x_int = torch.clamp(x_int, self.q_n, self.q_p)
        x_int = RoundSTE.apply(x_int)
        return x_int * s_safe


class QConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=True,
        weight_bitwidth=4,
        activation_bitwidth=4
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

        self.weight_quantizer = LSQQuantizer(
            bitwidth=weight_bitwidth,
            is_activation=False,
            per_channel=True,
            channels=out_channels
        )
        self.activation_quantizer = LSQQuantizer(
            bitwidth=activation_bitwidth,
            is_activation=True,
            per_channel=False
        )

    def forward(self, x):
        quantized_x = self.activation_quantizer(x)
        quantized_weight = self.weight_quantizer(self.conv.weight)
        return F.conv2d(
            quantized_x,
            quantized_weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.conv.groups
        )