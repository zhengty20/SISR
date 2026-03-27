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


class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class LSQPlusActQuant(nn.Module):
    def __init__(self, bitwidth=4, channels=1, signed=True, eps=1e-8):
        super().__init__()
        self.bitwidth = int(bitwidth)
        self.signed = bool(signed)
        self.eps = float(eps)
        self.disabled = self.bitwidth == -1 or self.bitwidth >= 32
        self.s = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.initialized = False

        if self.signed:
            self.qn = -float(1 << (self.bitwidth - 1))
            self.qp = float((1 << (self.bitwidth - 1)) - 1)
        else:
            self.qn = 0.0
            self.qp = float((1 << self.bitwidth) - 1)

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
        reduce_dims = tuple(i for i in range(x.dim()) if i != 1)
        x_mean = x.detach().mean(dim=reduce_dims, keepdim=True)
        x_std = x.detach().std(dim=reduce_dims, keepdim=True).clamp(min=self.eps)
        s_init = (2.0 * x_std) / (self.qp ** 0.5)
        self.s.data.copy_(s_init)
        self.beta.data.copy_(x_mean)
        self.initialized = True

    def forward(self, x, active_channels=None):
        if self.disabled:
            return x
        if not self.initialized:
            self._init_from_tensor(x)
        if active_channels is None:
            active_channels = x.shape[1]
        c = int(active_channels)
        s = self.s[:, :c]
        beta = self.beta[:, :c]
        n = x[:, :c].numel() / max(1, c)
        g = 1.0 / ((n * self.qp) ** 0.5)
        s = torch.clamp(ScaleGrad.apply(s, g), min=self.eps)
        x_hat = (x - beta) / s
        x_q = torch.clamp(RoundSTE.apply(x_hat), self.qn, self.qp)
        return x_q * s + beta


class LSQPlusWeightQuant(nn.Module):
    def __init__(self, bitwidth=4, out_channels=1, eps=1e-8):
        super().__init__()
        self.bitwidth = int(bitwidth)
        self.eps = float(eps)
        self.disabled = self.bitwidth == -1 or self.bitwidth >= 32
        self.qn = -float(1 << (self.bitwidth - 1))
        self.qp = float((1 << (self.bitwidth - 1)) - 1)
        self.s = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(out_channels, 1, 1, 1))
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

    def _init_from_tensor(self, w):
        if self.disabled:
            self.initialized = True
            return
        reduce_dims = tuple(range(1, w.dim()))
        w_mean = w.detach().mean(dim=reduce_dims, keepdim=True)
        w_std = w.detach().std(dim=reduce_dims, keepdim=True).clamp(min=self.eps)
        s_init = (2.0 * w_std) / (self.qp ** 0.5)
        self.s.data.copy_(s_init)
        self.beta.data.copy_(w_mean)
        self.initialized = True

    def forward(self, w, active_out_channels=None):
        if self.disabled:
            return w
        if not self.initialized:
            self._init_from_tensor(w)
        if active_out_channels is None:
            active_out_channels = w.shape[0]
        c = int(active_out_channels)
        s = self.s[:c]
        beta = self.beta[:c]
        n = w[0].numel()
        g = 1.0 / ((n * self.qp) ** 0.5)
        s = torch.clamp(ScaleGrad.apply(s, g), min=self.eps)
        w_hat = (w - beta) / s
        w_q = torch.clamp(RoundSTE.apply(w_hat), self.qn, self.qp)
        return w_q * s + beta


class QConv2dLSQP(nn.Module):
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
        self.act_quant = LSQPlusActQuant(
            bitwidth=activation_bitwidth,
            channels=in_channels,
            signed=act_signed,
        )
        self.weight_quant = LSQPlusWeightQuant(
            bitwidth=weight_bitwidth,
            out_channels=out_channels,
        )
        self.quantize_input = bool(quantize_input)

    def set_input_quantization(self, enabled: bool):
        self.quantize_input = bool(enabled)

    def _maybe_quantize_input(self, x):
        if self.quantize_input:
            return self.act_quant(x, active_channels=x.shape[1])
        return x

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

        raise NotImplementedError("Only groups=1 or depthwise are supported")

    def forward(self, x):
        qx = self._maybe_quantize_input(x)
        qw = self.weight_quant(self.conv.weight, active_out_channels=self.conv.out_channels)
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
        x_use, w_use, b_use, groups, active_out = self._slice_shared_tensors(
            x=x,
            active_in_channels=active_in_channels,
            active_out_channels=active_out_channels,
        )
        qx = self._maybe_quantize_input(x_use)
        qw = self.weight_quant(w_use, active_out_channels=active_out)
        return F.conv2d(
            qx,
            qw,
            b_use,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=groups,
        )
