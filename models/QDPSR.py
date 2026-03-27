import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from .QConv2d_RLQ import QConv2dRLQ
from .QConv2d_PACT_SAWB import QConv2dPACTSAWB
from .QConv2d_LSQP import QConv2dLSQP


def _build_qconv(qconv_cls, **kwargs):
    sig = inspect.signature(qconv_cls.__init__)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return qconv_cls(**valid)


class QBlock(nn.Module):
    def __init__(
        self,
        fea_dim,
        bias=False,
        weight_bitwidth=4,
        activation_bitwidth=4,
        qconv_cls=QConv2dRLQ,
        qconv_kwargs=None,
    ):
        super(QBlock, self).__init__()

        self.bias = bias
        self.fea_dim = fea_dim
        self.qconv_cls = qconv_cls
        self.qconv_kwargs = qconv_kwargs or {}

        self.projection1 = _build_qconv(
            self.qconv_cls,
            in_channels=fea_dim,
            out_channels=fea_dim,
            kernel_size=1,
            padding=0,
            bias=self.bias,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=activation_bitwidth,
            **self.qconv_kwargs,
        )
        self.projection2 = _build_qconv(
            self.qconv_cls,
            in_channels=fea_dim,
            out_channels=fea_dim,
            kernel_size=1,
            padding=0,
            bias=self.bias,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=activation_bitwidth,
            **self.qconv_kwargs,
        )
        self.filter1 = _build_qconv(
            self.qconv_cls,
            in_channels=fea_dim,
            out_channels=fea_dim,
            kernel_size=3,
            padding=1,
            bias=self.bias,
            groups=fea_dim,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=activation_bitwidth,
            **self.qconv_kwargs,
        )
        self.filter2 = _build_qconv(
            self.qconv_cls,
            in_channels=fea_dim,
            out_channels=fea_dim,
            kernel_size=3,
            padding=1,
            bias=self.bias,
            groups=fea_dim,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=activation_bitwidth,
            **self.qconv_kwargs,
        )
        self.act = nn.PReLU(num_parameters=self.fea_dim, init=0.25)
        self.scale = nn.Parameter(torch.ones(1, fea_dim, 1, 1))

    def forward(self, x):

        y = self.filter1(x)
        y = self.projection1(y)
        y = self.act(y)

        y = self.filter2(y)
        y = self.projection2(y)
        y = y * self.scale + x

        return y

    def forward_shared_channel(self, x, active_channels):
        c = int(active_channels)
        y = self.filter1.forward_shared_channel(x, c, c)
        y = self.projection1.forward_shared_channel(y, c, c)
        y = F.prelu(y, self.act.weight[:c])

        y = self.filter2.forward_shared_channel(y, c, c)
        y = self.projection2.forward_shared_channel(y, c, c)
        y = y * self.scale[:, :c] + x[:, :c]
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
        activation_bitwidth=4,
        head_tail_activation_bitwidth=8,
        qconv_cls=QConv2dRLQ,
        qconv_kwargs=None,
    ):
        super(QDPSR, self).__init__()

        self.scale = scale
        self.bias = bias
        self.fea_dim = fea_dim
        self.qconv_cls = qconv_cls
        self.qconv_kwargs = qconv_kwargs or {}

        self.head = _build_qconv(
            self.qconv_cls,
            in_channels=in_dim,
            out_channels=fea_dim,
            kernel_size=1,
            padding=0,
            bias=bias,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=head_tail_activation_bitwidth,
            **self.qconv_kwargs,
        )

        self.body = nn.ModuleList()
        for _ in range(num_blocks):
            self.body.append(
                QBlock(
                    fea_dim,
                    bias=bias,
                    weight_bitwidth=weight_bitwidth,
                    activation_bitwidth=activation_bitwidth,
                    qconv_cls=self.qconv_cls,
                    qconv_kwargs=self.qconv_kwargs,
                )
            )

        self.tail1 = _build_qconv(
            self.qconv_cls,
            in_channels=fea_dim,
            out_channels=fea_dim,
            kernel_size=3,
            padding=1,
            groups=fea_dim,
            bias=bias,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=head_tail_activation_bitwidth,
            **self.qconv_kwargs,
        )
        self.tail2 = _build_qconv(
            self.qconv_cls,
            in_channels=fea_dim,
            out_channels=in_dim * scale ** 2,
            kernel_size=1,
            padding=0,
            bias=bias,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=head_tail_activation_bitwidth,
            **self.qconv_kwargs,
        )
        
        self.upsampler = nn.PixelShuffle(scale)
        self.alpha = nn.Parameter(torch.ones(1, 3, 1, 1))

    def forward(self, x):

        y = self.head(x)

        for i in range(len(self.body)):
            y = self.body[i](y)

        y = self.tail1(y)
        y = self.tail2(y)
        y = self.alpha * self.upsampler(y)

        return y

    def forward_shared_channel(self, x, active_channels):
        c = max(1, min(int(active_channels), self.fea_dim))
        y = self.head.forward_shared_channel(x, self.head.conv.in_channels, c)

        for i in range(len(self.body)):
            y = self.body[i].forward_shared_channel(y, c)

        y = self.tail1.forward_shared_channel(y, c, c)
        y = self.tail2.forward_shared_channel(y, c, self.tail2.conv.out_channels)
        y = self.alpha * self.upsampler(y)
        return y

    def param_num(self):

        total = 0
        total += sum(p.numel() for p in self.head.conv.parameters())
        for i in range(len(self.body)):
            total += self.body[i].param_num()
        total += sum(p.numel() for p in self.tail1.conv.parameters())
        total += sum(p.numel() for p in self.tail2.conv.parameters())

        return total

def build_qdpsr(
    scale,
    in_dim,
    fea_dim,
    num_blocks=5,
    bias=False,
    weight_bitwidth=4,
    activation_bitwidth=4,
    head_tail_activation_bitwidth=8,
    quant_method="rlq",
):
    quant_method = str(quant_method).lower()
    if quant_method == "rlq":
        qconv_cls = QConv2dRLQ
        qconv_kwargs = {}
    elif quant_method == "pact_sawb":
        qconv_cls = QConv2dPACTSAWB
        qconv_kwargs = {}
    elif quant_method == "lsq_plus":
        qconv_cls = QConv2dLSQP
        qconv_kwargs = {}
    else:
        raise ValueError(f"不支持的 quant_method: {quant_method}")

    return QDPSR(
        scale=scale,
        in_dim=in_dim,
        fea_dim=fea_dim,
        num_blocks=num_blocks,
        bias=bias,
        weight_bitwidth=weight_bitwidth,
        activation_bitwidth=activation_bitwidth,
        head_tail_activation_bitwidth=head_tail_activation_bitwidth,
        qconv_cls=qconv_cls,
        qconv_kwargs=qconv_kwargs,
    )