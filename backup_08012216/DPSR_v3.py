import torch.nn.functional as F
import torch.nn as nn


class USConv2d(nn.Conv2d):
    """A two-width convolution that slices weights from the full-width layer."""

    def __init__(self, *args, us=(False, False), **kwargs):
        super().__init__(*args, **kwargs)
        self.us = tuple(us)

    def forward(self, x, width_mult=1.0):
        in_channels = x.shape[1] if self.us[0] else self.in_channels
        out_channels = int(self.out_channels * width_mult) if self.us[1] else self.out_channels

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

        self.projection1 = USConv2d(
            fea_dim, fea_dim, kernel_size=1, padding=0, bias=self.bias, us=(False, True)
        )
        self.projection2 = USConv2d(
            fea_dim, fea_dim, kernel_size=1, padding=0, bias=self.bias, us=(True, False)
        )
        self.filter1 = nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=self.bias, groups=fea_dim)
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
        self.act2 = nn.PReLU(num_parameters=self.fea_dim, init=0.25)

    def forward(self, x, width_mult=1.0):
        y = self.filter1(x)
        y = self.projection1(y, width_mult=width_mult)
        y = self.act1(y)
        
        y = self.filter2(y, width_mult=width_mult)
        y = self.projection2(y, width_mult=width_mult)
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

    def __init__(self, scale, in_dim, fea_dim, num_blocks=5, bias=True):
        super(DPSR, self).__init__()

        if fea_dim < 2:
            raise ValueError("fea_dim must be at least 2 for the half-width subnet")

        self.scale = scale
        self.bias = bias
        self.fea_dim = fea_dim

        self.head = nn.Conv2d(in_dim, fea_dim, kernel_size=1, padding=0, bias=bias)

        self.body = nn.ModuleList()
        for _ in range(num_blocks):
            self.body.append(Block(fea_dim, bias=bias))

        # self.tail1 = nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=bias, groups=fea_dim)
        # self.tail2 = nn.Conv2d(fea_dim, in_dim * scale ** 2, kernel_size=1, padding=0, bias=bias)
        self.tail = nn.Conv2d(fea_dim, in_dim * scale ** 2, kernel_size=1, padding=0, bias=bias)

        self.upsampler = nn.PixelShuffle(scale)
        
    def forward(self, x, width_mult=1.0):
        width_mult = float(width_mult)
        if width_mult not in self.supported_width_mults:
            raise ValueError(
                f"width_mult must be one of {self.supported_width_mults}, got {width_mult}"
            )
        
        y = self.head(x)
        
        for i in range(len(self.body)):
            y = self.body[i](y, width_mult=width_mult)

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