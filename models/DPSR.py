import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, fea_dim, bias = True):
        super(Block, self).__init__()
        
        self.bias = bias
        self.fea_dim = fea_dim

        self.projection1 = nn.Conv2d(fea_dim, fea_dim, kernel_size=1, padding=0, bias=self.bias)
        self.filter1 = nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=self.bias, groups=fea_dim)
        self.projection2 = nn.Conv2d(fea_dim, fea_dim, kernel_size=1, padding=0, bias=self.bias)
        self.filter2 = nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=self.bias, groups=fea_dim)

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
        total += sum(p.numel() for p in self.projection1.parameters())
        total += sum(p.numel() for p in self.filter1.parameters())
        total += sum(p.numel() for p in self.projection2.parameters())
        total += sum(p.numel() for p in self.filter2.parameters())

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
        
        y = self.head(x)
        
        for i in range(len(self.body)):
            y = self.body[i](y)

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