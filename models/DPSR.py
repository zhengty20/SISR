import torch.nn as nn
import torch.nn.functional as F


def channel_label(channels):
    return f"{int(channels)}ch"


class USConv2d(nn.Conv2d):
    """Convolution whose active input/output channels slice full-width weights."""

    def __init__(self, *args, us=(False, False), **kwargs):
        super().__init__(*args, **kwargs)
        self.us = tuple(us)

    def forward(self, x, active_out_channels=None):
        in_channels = x.shape[1] if self.us[0] else self.in_channels
        out_channels = (
            int(active_out_channels)
            if active_out_channels is not None
            else self.out_channels
        )
        if not 0 < in_channels <= self.in_channels:
            raise ValueError(
                f"active input channels must be in [1, {self.in_channels}], got {in_channels}"
            )
        if not 0 < out_channels <= self.out_channels:
            raise ValueError(
                f"active output channels must be in [1, {self.out_channels}], got {out_channels}"
            )
        if self.groups == 1:
            weight, groups = self.weight[:out_channels, :in_channels], 1
        else:
            if not (self.groups == self.in_channels == self.out_channels):
                raise ValueError(
                    "Dynamic grouped convolution only supports depthwise convolution"
                )
            if in_channels != out_channels:
                raise ValueError("Depthwise input and output channels must match")
            weight, groups = self.weight[:out_channels], in_channels
        bias = self.bias[:out_channels] if self.bias is not None else None
        return F.conv2d(
            x, weight, bias, self.stride, self.padding, self.dilation, groups
        )


class USPReLU(nn.PReLU):
    """PReLU that shares the leading parameters with the subnet."""

    def forward(self, x):
        return F.prelu(x, self.weight[: x.shape[1]])


class Block(nn.Module):
    def __init__(self, fea_dim, subnet_channels, bias=True):
        super().__init__()
        self.supported_channels = (subnet_channels, fea_dim)
        kwargs = dict(kernel_size=1, padding=0, bias=bias, us=(True, True))
        self.projection1 = USConv2d(fea_dim, fea_dim, **kwargs)
        self.projection2 = USConv2d(fea_dim, fea_dim, **kwargs)
        depthwise_kwargs = dict(
            kernel_size=3,
            padding=1,
            bias=bias,
            groups=fea_dim,
            us=(True, True),
        )
        self.filter1 = USConv2d(fea_dim, fea_dim, **depthwise_kwargs)
        self.filter2 = USConv2d(fea_dim, fea_dim, **depthwise_kwargs)
        self.act1 = USPReLU(num_parameters=fea_dim, init=0.25)
        self.act2 = USPReLU(num_parameters=fea_dim, init=0.25)

    def forward(self, x):
        channels = x.shape[1]
        if channels not in self.supported_channels:
            raise ValueError(
                f"block input channels must be one of {self.supported_channels}, got {channels}"
            )
        y = self.act1(
            self.projection1(
                self.filter1(x, active_out_channels=channels),
                active_out_channels=channels,
            )
        )
        return self.act2(
            self.projection2(
                self.filter2(y, active_out_channels=channels),
                active_out_channels=channels,
            )
        )


class DPSR(nn.Module):
    def __init__(
        self, scale, in_dim, fea_dim, num_blocks=5, bias=True, subnet_channels=16
    ):
        super().__init__()
        if not 0 < subnet_channels < fea_dim:
            raise ValueError(
                f"subnet_channels must be in [1, {fea_dim - 1}], got {subnet_channels}"
            )
        self.scale = scale
        self.fea_dim = int(fea_dim)
        self.subnet_channels = int(subnet_channels)
        self.num_blocks = int(num_blocks)
        self.head = USConv2d(
            in_dim, fea_dim, kernel_size=1, padding=0, bias=bias, us=(False, True)
        )
        self.body = nn.ModuleList(
            Block(fea_dim, self.subnet_channels, bias=bias) for _ in range(num_blocks)
        )
        self.tail = USConv2d(
            fea_dim,
            in_dim * scale**2,
            kernel_size=1,
            padding=0,
            bias=bias,
            us=(True, False),
        )
        self.upsampler = nn.PixelShuffle(scale)

    @property
    def supported_channels(self):
        return (self.fea_dim, self.subnet_channels)

    def forward(self, x, channels=None):
        channels = self.fea_dim if channels is None else int(channels)
        if channels not in self.supported_channels:
            raise ValueError(
                f"channels must be one of {self.supported_channels}, got {channels}"
            )
        y = self.head(x, active_out_channels=channels)
        for block in self.body:
            y = block(y)
        return self.upsampler(self.tail(y))

    def param_num(self):
        return sum(parameter.numel() for parameter in self.parameters())


if __name__ == "__main__":
    model = DPSR(2, 3, 32, 5, bias=False, subnet_channels=16)
    print(f"参数数量: {model.param_num()}")
