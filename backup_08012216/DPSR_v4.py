import torch.nn.functional as F
import torch.nn as nn


class USConv2d(nn.Conv2d):
    """A two-width convolution that slices weights from the full-width layer."""

    def __init__(self, *args, us=(False, False), **kwargs):
        super().__init__(*args, **kwargs)
        self.us = tuple(us)

    def forward(self, x, width_mult=1.0, active_out_channels=None):
        in_channels = x.shape[1] if self.us[0] else self.in_channels
        if active_out_channels is not None:
            out_channels = int(active_out_channels)
        elif self.us[1]:
            out_channels = int(self.out_channels * width_mult)
        else:
            out_channels = self.out_channels

        if not 0 < in_channels <= self.in_channels:
            raise ValueError(
                f"active input channels must be in [1, {self.in_channels}], got {in_channels}"
            )
        if not 0 < out_channels <= self.out_channels:
            raise ValueError(
                f"active output channels must be in [1, {self.out_channels}], got {out_channels}"
            )

        if self.groups == 1:
            weight = self.weight[:out_channels, :in_channels]
            groups = 1
        else:
            if not (self.groups == self.in_channels == self.out_channels):
                raise ValueError("Dynamic grouped convolution only supports depthwise convolution")
            if in_channels != out_channels:
                raise ValueError("Depthwise input and output channels must match")
            weight = self.weight[:out_channels]
            groups = in_channels

        bias = self.bias[:out_channels] if self.bias is not None else None
        return F.conv2d(
            x,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            groups,
        )


class USPReLU(nn.PReLU):
    """PReLU sharing the first parameters with the half-width subnet."""

    def forward(self, x):
        return F.prelu(x, self.weight[:x.shape[1]])


class Block(nn.Module):
    def __init__(self, fea_dim, bias = True):
        super(Block, self).__init__() 
        self.bias = bias
        self.fea_dim = fea_dim
        self.half_fea_dim = fea_dim // 2

        self.projection1 = USConv2d(
            fea_dim, fea_dim, kernel_size=1, padding=0, bias=self.bias, us=(True, True)
        )
        self.projection2 = USConv2d(
            fea_dim, fea_dim, kernel_size=1, padding=0, bias=self.bias, us=(True, True)
        )
        self.filter1 = USConv2d(
            fea_dim,
            fea_dim,
            kernel_size=3,
            padding=1,
            bias=self.bias,
            groups=fea_dim,
            us=(True, True),
        )
        self.filter2 = USConv2d(
            fea_dim,
            fea_dim,
            kernel_size=3,
            padding=1,
            bias=self.bias,
            groups=fea_dim,
            us=(True, True),
        )

        self.act1 = USPReLU(num_parameters=self.fea_dim, init=0.25)
        self.act2 = USPReLU(num_parameters=self.fea_dim, init=0.25)

    def forward(self, x, output_channels=None):
        active_channels = x.shape[1]
        if active_channels not in (self.half_fea_dim, self.fea_dim):
            raise ValueError(
                f"block input channels must be {self.half_fea_dim} or {self.fea_dim}, "
                f"got {active_channels}"
            )
        if output_channels is None:
            output_channels = active_channels
        output_channels = int(output_channels)
        if output_channels not in (self.half_fea_dim, self.fea_dim):
            raise ValueError(
                f"block output channels must be {self.half_fea_dim} or {self.fea_dim}, "
                f"got {output_channels}"
            )

        y = self.filter1(x, active_out_channels=active_channels)
        y = self.projection1(y, active_out_channels=active_channels)
        y = self.act1(y)

        y = self.filter2(y, active_out_channels=active_channels)
        y = self.projection2(y, active_out_channels=output_channels)
        y = self.act2(y)

        return y

    def param_num(self):      
        total = 0
        total += sum(p.numel() for p in self.projection1.parameters())
        total += sum(p.numel() for p in self.filter1.parameters())
        total += sum(p.numel() for p in self.projection2.parameters())
        total += sum(p.numel() for p in self.filter2.parameters())

        return total

class DPSR(nn.Module):
    supported_width_mults = (1.0, 0.5)

    def __init__(
        self, scale, in_dim, fea_dim, num_blocks=5, bias=True, subnet_expand_block=3
    ):
        super(DPSR, self).__init__()

        if fea_dim < 2:
            raise ValueError("fea_dim must be at least 2 for the half-width subnet")
        if not 1 <= int(subnet_expand_block) <= num_blocks:
            raise ValueError(
                f"subnet_expand_block must be in [1, {num_blocks}], got {subnet_expand_block}"
            )

        self.scale = scale
        self.bias = bias
        self.fea_dim = fea_dim

        self.half_fea_dim = fea_dim // 2
        self.num_blocks = int(num_blocks)
        self.subnet_expand_block = int(subnet_expand_block)
        self.head = USConv2d(
            in_dim, fea_dim, kernel_size=1, padding=0, bias=bias, us=(False, True)
        )

        self.body = nn.ModuleList()
        for _ in range(num_blocks):
            self.body.append(Block(fea_dim, bias=bias))

        # self.tail1 = nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=bias, groups=fea_dim)
        # self.tail2 = nn.Conv2d(fea_dim, in_dim * scale ** 2, kernel_size=1, padding=0, bias=bias)
        self.tail = nn.Conv2d(fea_dim, in_dim * scale ** 2, kernel_size=1, padding=0, bias=bias)

        self.upsampler = nn.PixelShuffle(scale)
        
    def forward(self, x, width_mult=1.0, subnet_expand_block=None):
        width_mult = float(width_mult)
        if width_mult not in self.supported_width_mults:
            raise ValueError(
                f"width_mult must be one of {self.supported_width_mults}, got {width_mult}"
            )

        if subnet_expand_block is None:
            subnet_expand_block = self.subnet_expand_block
        subnet_expand_block = int(subnet_expand_block)
        if not 1 <= subnet_expand_block <= self.num_blocks:
            raise ValueError(
                f"subnet_expand_block must be in [1, {self.num_blocks}], "
                f"got {subnet_expand_block}"
            )

        if width_mult == 1.0:
            y = self.head(x, width_mult=1.0)
            for block in self.body:
                y = block(y, output_channels=self.fea_dim)
        else:
            y = self.head(x, width_mult=0.5)
            for block_index, block in enumerate(self.body, start=1):
                output_channels = (
                    self.half_fea_dim
                    if block_index < subnet_expand_block
                    else self.fea_dim
                )
                y = block(y, output_channels=output_channels)

        # y = self.tail1(y)
        # y = self.tail2(y)
        y = self.tail(y)
        y = self.upsampler(y)

        return y
    
    def param_num(self):
        
        total = 0
        total += sum(p.numel() for p in self.head.parameters()) 

        for i in range(len(self.body)):
            total += self.body[i].param_num()

        # total += sum(p.numel() for p in self.tail1.parameters())
        # total += sum(p.numel() for p in self.tail2.parameters())
        total += sum(p.numel() for p in self.tail.parameters())

        return total

if __name__ == '__main__':
    
    model = DPSR(2, 3, 32, 5, bias=False)
    print(f"参数数量: {model.param_num()}")