import torch
import torch.nn as nn
from .QConv2d import QConv2d

class QBlock(nn.Module):
    
    def __init__(self, in_dim, out_dim, bias = True):
        super(QBlock, self).__init__()
        
        self.bias = bias
        self.in_dim = in_dim

        self.filter1 = QConv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=bias, groups=in_dim, weight_bitwidth=4, activation_bitwidth=8)
        self.filter2 = QConv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=bias, groups=in_dim, weight_bitwidth=4, activation_bitwidth=8)
        self.projection1 = QConv2d(in_dim, out_dim, kernel_size=1, padding=0, bias=bias, weight_bitwidth=4, activation_bitwidth=8)
        self.projection2 = QConv2d(in_dim, out_dim, kernel_size=1, padding=0, bias=bias, weight_bitwidth=4, activation_bitwidth=8)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):

        y = self.filter1(x)
        y = self.projection1(y)
        y = self.act(y)
        
        y = self.filter2(y)
        y = self.projection2(y)
        y = self.act(y) + x

        return y
    
    def param_num(self):
        
        total = 0
        total += sum(p.numel() for p in self.filter1.conv.parameters())
        total += sum(p.numel() for p in self.filter2.conv.parameters())
        total += sum(p.numel() for p in self.projection1.conv.parameters())
        total += sum(p.numel() for p in self.projection2.conv.parameters())

        return total

class QISCSR(nn.Module):

    def __init__(self, scale, in_dim, fea_dim, num_blocks=2, bias=True):
        super(QISCSR, self).__init__()

        self.scale = scale
        self.fea_dim = fea_dim
        self.head = QConv2d(in_dim, fea_dim, kernel_size=3, padding=1, bias=bias, weight_bitwidth=4, activation_bitwidth=8)
        
        self.body = nn.ModuleList()
        for i in range(num_blocks):
            self.body.append(QBlock(fea_dim, fea_dim, bias=bias))

        self.tail = QConv2d(fea_dim, in_dim * scale ** 2, kernel_size=3, padding=1, bias=bias, weight_bitwidth=4, activation_bitwidth=8)
        # self.tail1 = QConv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=bias, groups=fea_dim, weight_bitwidth=4, activation_bitwidth=4)
        # self.tail2 = QConv2d(fea_dim, in_dim * scale ** 2, kernel_size=1, padding=0, bias=bias, weight_bitwidth=4, activation_bitwidth=4)
       
        self.upsampler = nn.PixelShuffle(scale)
        self.alpha = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        
        y = self.head(x)
        y = self.act(y)
        
        for i in range(len(self.body)):
            y = self.body[i](y)

        y = self.tail(y)
        # y = self.tail2(y)
        y = self.alpha * self.upsampler(y) 

        return y
    
    def param_num(self):
        
        total = 0
        total += sum(p.numel() for p in self.head.conv.parameters())
        for i in range(len(self.body)):
            total += self.body[i].param_num()
        total += sum(p.numel() for p in self.tail.conv.parameters())
        # total += sum(p.numel() for p in self.tail2.conv.parameters())

        return total