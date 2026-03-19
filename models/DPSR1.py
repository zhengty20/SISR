import torch
import torch.nn as nn

class ECALayer(nn.Module):
    
    def __init__(self):
        super(ECALayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.gate(y)

    def param_num(self):
        return sum(p.numel() for p in self.conv.parameters())

class Block(nn.Module):
    
    def __init__(self, fea_dim, bias = True):
        super(Block, self).__init__()
        
        self.bias = bias
        self.fea_dim = fea_dim
        self.hidden_dim = fea_dim + fea_dim // 2

        self.expand = nn.Conv2d(fea_dim, self.hidden_dim, kernel_size=1, padding=0, bias=self.bias)
        self.filter = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, bias=self.bias, groups=self.hidden_dim)
        self.project = nn.Conv2d(self.hidden_dim, fea_dim, kernel_size=1, padding=0, bias=self.bias)

        self.act = nn.PReLU(num_parameters=self.hidden_dim, init=0.25)
        self.attn = ECALayer()

    def forward(self, x):

        y = self.expand(x)
        y = self.filter(y)
        y = self.act(y)
        y = self.attn(y)
        y = self.project(y) + x

        return y

    def param_num(self):
        
        total = 0
        total += sum(p.numel() for p in self.expand.parameters())
        total += sum(p.numel() for p in self.filter.parameters())
        total += sum(p.numel() for p in self.project.parameters())
        total += self.attn.param_num()

        return total

class DPSR(nn.Module):

    def __init__(self, scale, in_dim, fea_dim, num_blocks=2, bias=True):
        super(DPSR, self).__init__()

        self.scale = scale
        self.bias = bias

        self.head = nn.Conv2d(in_dim, fea_dim, kernel_size=3, padding=1, bias=bias)

        self.body = nn.ModuleList()
        for _ in range(num_blocks):
            self.body.append(Block(fea_dim, bias=bias))

        self.tail1 = nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=bias, groups=fea_dim)
        self.tail2 = nn.Conv2d(fea_dim, in_dim * scale ** 2, kernel_size=1, padding=0, bias=bias)

        self.upsampler = nn.PixelShuffle(scale)
        self.alpha = nn.Parameter(torch.ones(1, 3, 1, 1))
        
    def forward(self, x):
        
        shallow = self.head(x)
        y = shallow
        
        for i in range(len(self.body)):
            y = self.body[i](y)

        y = y + shallow
        y = self.tail1(y)
        y = self.tail2(y)
        y = self.alpha * self.upsampler(y)

        return y
    
    def param_num(self):
        
        total = 0
        total += sum(p.numel() for p in self.head.parameters())

        for i in range(len(self.body)):
            total += self.body[i].param_num()

        total += sum(p.numel() for p in self.tail1.parameters())
        total += sum(p.numel() for p in self.tail2.parameters())

        return total

if __name__ == '__main__':
    
    model = DPSR(2, 3, 32, 4, bias=False)
    print(f"参数数量: {model.param_num()}")