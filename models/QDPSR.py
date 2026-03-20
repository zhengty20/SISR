import torch
import torch.nn as nn
from .QConv2d import QConv2d


class QBlock(nn.Module):

    def __init__(
        self,
        fea_dim,
        bias=False,
        weight_bitwidth=4,
        activation_bitwidth=8
    ):
        super(QBlock, self).__init__()

        self.bias = bias
        self.fea_dim = fea_dim

        self.projection1 = QConv2d(
            fea_dim,
            fea_dim,
            kernel_size=1,
            padding=0,
            bias=self.bias,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=activation_bitwidth
        )
        self.filter1 = QConv2d(
            fea_dim,
            fea_dim,
            kernel_size=3,
            padding=1,
            bias=self.bias,
            groups=fea_dim,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=activation_bitwidth
        )
        self.projection2 = QConv2d(
            fea_dim,
            fea_dim,
            kernel_size=1,
            padding=0,
            bias=self.bias,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=activation_bitwidth
        )
        self.filter2 = QConv2d(
            fea_dim,
            fea_dim,
            kernel_size=3,
            padding=1,
            bias=self.bias,
            groups=fea_dim,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=activation_bitwidth
        )
        self.act1 = nn.PReLU(num_parameters=self.fea_dim, init=0.25)
        self.act2 = nn.PReLU(num_parameters=self.fea_dim, init=0.25)

    def forward(self, x):

        y = self.projection1(x)
        y = self.filter1(y)
        y = self.act1(y)

        y = self.projection2(y)
        y = self.filter2(y)
        y = self.act2(y) + x

        return y

    def param_num(self):

        total = 0
        total += sum(p.numel() for p in self.projection1.conv.parameters())
        total += sum(p.numel() for p in self.filter1.conv.parameters())
        total += sum(p.numel() for p in self.projection2.conv.parameters())
        total += sum(p.numel() for p in self.filter2.conv.parameters())

        return total

class QDPSR(nn.Module):

    def __init__(
        self,
        scale,
        in_dim,
        fea_dim,
        num_blocks=5,
        bias=False,
        weight_bitwidth=4,
        activation_bitwidth=4
    ):
        super(QDPSR, self).__init__()

        self.scale = scale
        self.bias = bias
        self.fea_dim = fea_dim
        self.head = QConv2d(
            in_dim,
            fea_dim,
            kernel_size=3,
            padding=1,
            bias=bias,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=8
        )

        self.body = nn.ModuleList()
        for _ in range(num_blocks):
            self.body.append(
                QBlock(
                    fea_dim,
                    bias=bias,
                    weight_bitwidth=weight_bitwidth,
                    activation_bitwidth=activation_bitwidth
                )
            )

        self.tail = QConv2d(
            fea_dim,
            in_dim * scale ** 2,
            kernel_size=3,
            padding=1,
            bias=bias,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=8
        )

        self.upsampler = nn.PixelShuffle(scale)
        self.alpha = nn.Parameter(torch.ones(1, 3, 1, 1))

    def forward(self, x):

        y = self.head(x)

        for i in range(len(self.body)):
            y = self.body[i](y)

        y = self.tail(y)
        y = self.alpha * self.upsampler(y)

        return y

    def param_num(self):

        total = 0
        total += sum(p.numel() for p in self.head.conv.parameters())
        for i in range(len(self.body)):
            total += self.body[i].param_num()
        total += sum(p.numel() for p in self.tail.conv.parameters())

        return total