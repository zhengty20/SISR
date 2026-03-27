import math

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


class RLQQuantizer(nn.Module):
    """Redistribution-driven learnable quantizer (RLQ-style)."""

    def __init__(self, bitwidth, is_activation=False, channels=None, eps=1e-8):
        super().__init__()
        if bitwidth < 2 and bitwidth != -1:
            raise ValueError("bitwidth 必须 >= 2，或使用 -1 表示关闭量化")

        self.bitwidth = bitwidth
        self.is_activation = is_activation
        self.eps = eps
        self.disabled = bitwidth == -1 or bitwidth >= 32
        self.q_n = -2 ** (bitwidth - 1) if not self.disabled else None
        self.q_p = 2 ** (bitwidth - 1) - 1 if not self.disabled else None
        self._tanh_one = math.tanh(1.0)

        if self.disabled:
            self.register_parameter("s", None)
            self.register_parameter("tau", None)
            self.initialized = True
            return

        if channels is None:
            raise ValueError("量化器必须提供 channels 以支持 per-channel scale")
        if self.is_activation:
            self.s = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.tau = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            self.s = nn.Parameter(torch.ones(channels, 1, 1, 1))
            self.tau = nn.Parameter(torch.zeros(channels, 1, 1, 1))
        self.initialized = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self.initialized = True

    def _init_from_tensor(self, x):
        if self.disabled:
            self.initialized = True
            return

        max_abs = x.detach().abs()
        if self.is_activation:
            if x.dim() < 2:
                raise ValueError("激活量化输入维度必须 >= 2 且包含 channel 维")
            reduce_dims = tuple(i for i in range(x.dim()) if i != 1)
            channel_max = max_abs.amax(dim=reduce_dims, keepdim=True)
            s_init = channel_max / max(1.0, float(self.q_p))
            self.tau.data.zero_()
        else:
            reduce_dims = tuple(range(1, x.dim()))
            channel_max = max_abs.amax(dim=reduce_dims, keepdim=True)
            s_init = channel_max / max(1.0, float(self.q_p))
            self.tau.data.zero_()
        s_init = torch.clamp(s_init, min=self.eps)
        self.s.data.copy_(s_init.view_as(self.s))
        self.initialized = True

    def _rlq_transform(self, x_scaled):
        floor_detached = torch.floor(x_scaled).detach()
        frac = x_scaled - floor_detached
        phi = torch.tanh(2.0 * frac - 1.0) / self._tanh_one
        return floor_detached + 0.5 * (phi + 1.0)

    def _quantize_impl(self, x, s, tau):
        if self.disabled:
            return x

        x_shifted = x + tau
        if self.is_activation:
            reduce_dims = tuple(i for i in range(x_shifted.dim()) if i != 1)
            clip_bound = x_shifted.detach().abs().amax(dim=reduce_dims, keepdim=True)
        else:
            reduce_dims = tuple(range(1, x_shifted.dim()))
            clip_bound = x_shifted.detach().abs().amax(dim=reduce_dims, keepdim=True)
        clip_bound = torch.clamp(clip_bound, min=self.eps)
        x_clipped = torch.clamp(x_shifted, -clip_bound, clip_bound)

        x_scaled = x_clipped / s
        x_redistributed = self._rlq_transform(x_scaled)
        x_int = RoundSTE.apply(x_redistributed)
        x_int = torch.clamp(x_int, self.q_n, self.q_p)
        return x_int * s

    def forward(self, x, active_channels=None):
        if self.disabled:
            return x
        if not self.initialized:
            self._init_from_tensor(x)

        if self.is_activation:
            if active_channels is None:
                active_channels = x.shape[1]
            active_channels = int(active_channels)
            s = self.s[:, :active_channels]
            tau = self.tau[:, :active_channels]
            n = x[:, :active_channels].numel() / max(1, active_channels)
        else:
            if active_channels is None:
                active_channels = x.shape[0]
            active_channels = int(active_channels)
            s = self.s[:active_channels]
            tau = self.tau[:active_channels]
            n = x[0].numel()

        grad_scale = 1.0 / ((n * self.q_p) ** 0.5)
        s_scaled = ScaleGradient.apply(s, grad_scale)
        s_safe = torch.clamp(s_scaled, min=self.eps)
        return self._quantize_impl(x, s_safe, tau)


class QConv2dRLQ(nn.Module):
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

        self.weight_quantizer = RLQQuantizer(
            bitwidth=weight_bitwidth,
            is_activation=False,
            channels=out_channels,
        )
        self.activation_quantizer = RLQQuantizer(
            bitwidth=activation_bitwidth,
            is_activation=True,
            channels=in_channels,
        )
        self.quantize_input = bool(quantize_input)

    def set_input_quantization(self, enabled: bool):
        self.quantize_input = bool(enabled)

    def _maybe_quantize_input(self, x):
        if self.quantize_input:
            return self.activation_quantizer(x, active_channels=x.shape[1])
        return x

    def _quantize_weight_slice(self, weight, active_out_channels):
        quantizer = self.weight_quantizer
        if quantizer.disabled:
            return weight
        if not quantizer.initialized:
            quantizer._init_from_tensor(self.conv.weight)

        active_out_channels = int(active_out_channels)
        if active_out_channels <= 0:
            raise ValueError("active_out_channels 必须大于 0")

        return quantizer(weight, active_channels=active_out_channels)

    def _slice_shared_tensors(self, x, active_in_channels, active_out_channels):
        if self.conv.groups == 1:
            x_use = x[:, :active_in_channels]
            w_use = self.conv.weight[:active_out_channels, :active_in_channels]
            b_use = self.conv.bias[:active_out_channels] if self.conv.bias is not None else None
            return x_use, w_use, b_use, 1, active_out_channels

        is_depthwise = self.conv.groups == self.conv.in_channels and self.conv.in_channels == self.conv.out_channels
        if is_depthwise:
            active_c = min(active_in_channels, active_out_channels)
            x_use = x[:, :active_c]
            w_use = self.conv.weight[:active_c]
            b_use = self.conv.bias[:active_c] if self.conv.bias is not None else None
            return x_use, w_use, b_use, active_c, active_c

        raise NotImplementedError("当前仅支持 groups=1 或 depthwise 的 shared_channel 前向")

    def forward(self, x):
        quantized_x = self._maybe_quantize_input(x)
        quantized_weight = self._quantize_weight_slice(self.conv.weight, self.conv.out_channels)
        return F.conv2d(
            quantized_x,
            quantized_weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.conv.groups,
        )

    def forward_shared_channel(self, x, active_in_channels, active_out_channels):
        active_in_channels = int(active_in_channels)
        active_out_channels = int(active_out_channels)
        x_use, w_use, b_use, groups, active_out = self._slice_shared_tensors(
            x=x,
            active_in_channels=active_in_channels,
            active_out_channels=active_out_channels,
        )
        quantized_x = self._maybe_quantize_input(x_use)
        quantized_weight = self._quantize_weight_slice(w_use, active_out)
        return F.conv2d(
            quantized_x,
            quantized_weight,
            b_use,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=groups,
        )
