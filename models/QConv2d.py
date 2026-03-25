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
    def __init__(self, bitwidth, is_activation=False, channels=None, eps=1e-8):
        super().__init__()
        if bitwidth < 2:
            raise ValueError("LSQ bitwidth must be >= 2")

        self.bitwidth = bitwidth
        self.is_activation = is_activation
        self.q_n = -2 ** (bitwidth - 1)
        self.q_p = 2 ** (bitwidth - 1) - 1
        self.eps = eps
        if self.is_activation:
            self.s = nn.Parameter(torch.tensor(1.0))
        else:
            if channels is None:
                raise ValueError("权重量化必须提供 channels")
            self.s = nn.Parameter(torch.ones(channels, 1, 1, 1))
        self.beta = nn.Parameter(torch.tensor(0.0)) if self.is_activation else None
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
        # If quantization parameters are loaded from checkpoint, keep them and skip runtime re-init.
        self.initialized = True

    def _init_from_tensor(self, x):
        if self.is_activation:
            x_min = x.detach().min()
            x_max = x.detach().max()
            s_init = (x_max - x_min) / max(1.0, float(self.q_p - self.q_n))
            s_init = torch.clamp(s_init, min=self.eps)
            beta_init = x_min - s_init * self.q_n
            self.beta.data.copy_(beta_init)
        else:
            reduce_dims = tuple(range(1, x.dim()))
            mean_abs = x.detach().abs().mean(dim=reduce_dims, keepdim=True)
            s_init = (2.0 * mean_abs) / (self.q_p ** 0.5)
        s_init = torch.clamp(s_init, min=self.eps)
        if self.is_activation:
            self.s.data.copy_(s_init)
        else:
            self.s.data.copy_(s_init.view_as(self.s))
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self._init_from_tensor(x)

        n = x.numel() if self.is_activation else x[0].numel()
        grad_scale = 1.0 / ((n * self.q_p) ** 0.5)
        s_scaled = ScaleGradient.apply(self.s, grad_scale)
        s_safe = torch.clamp(s_scaled, min=self.eps)
        if self.is_activation:
            beta_scaled = self.beta
            x_int = (x - beta_scaled) / s_safe
            x_int = torch.clamp(x_int, self.q_n, self.q_p)
            x_int = RoundSTE.apply(x_int)
            return x_int * s_safe + beta_scaled
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
        bias=False,
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
            channels=out_channels
        )
        self.activation_quantizer = LSQQuantizer(
            bitwidth=activation_bitwidth,
            is_activation=True
        )

    def _quantize_weight_slice(self, weight, active_out_channels):
        quantizer = self.weight_quantizer
        if not quantizer.initialized:
            # Always initialize from full tensor for stable per-channel scales.
            quantizer._init_from_tensor(self.conv.weight)

        active_out_channels = int(active_out_channels)
        if active_out_channels <= 0:
            raise ValueError("active_out_channels 必须大于 0")

        n = weight[0].numel()
        grad_scale = 1.0 / ((n * quantizer.q_p) ** 0.5)
        s = quantizer.s[:active_out_channels]
        s_scaled = ScaleGradient.apply(s, grad_scale)
        s_safe = torch.clamp(s_scaled, min=quantizer.eps)
        x_int = weight / s_safe
        x_int = torch.clamp(x_int, quantizer.q_n, quantizer.q_p)
        x_int = RoundSTE.apply(x_int)
        return x_int * s_safe

    def forward(self, x):
        quantized_x = self.activation_quantizer(x)
        quantized_weight = self._quantize_weight_slice(self.conv.weight, self.conv.out_channels)
        return F.conv2d(
            quantized_x,
            quantized_weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.conv.groups
        )

    def forward_shared_channel(self, x, active_in_channels, active_out_channels):
        active_in_channels = int(active_in_channels)
        active_out_channels = int(active_out_channels)

        if self.conv.groups == 1:
            x_use = x[:, :active_in_channels]
            w_use = self.conv.weight[:active_out_channels, :active_in_channels]
            b_use = self.conv.bias[:active_out_channels] if self.conv.bias is not None else None
            groups = 1
            active_out = active_out_channels
        elif self.conv.groups == self.conv.in_channels and self.conv.in_channels == self.conv.out_channels:
            # Depthwise convolution: enforce matched active channels.
            active_c = min(active_in_channels, active_out_channels)
            x_use = x[:, :active_c]
            w_use = self.conv.weight[:active_c]
            b_use = self.conv.bias[:active_c] if self.conv.bias is not None else None
            groups = active_c
            active_out = active_c
        else:
            raise NotImplementedError("当前仅支持 groups=1 或 depthwise 的 shared_channel 前向")

        quantized_x = self.activation_quantizer(x_use)
        quantized_weight = self._quantize_weight_slice(w_use, active_out)
        return F.conv2d(
            quantized_x,
            quantized_weight,
            b_use,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=groups
        )