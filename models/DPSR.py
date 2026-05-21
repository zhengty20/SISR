import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, fea_dim, bias = True):
        super(Block, self).__init__()
        
        self.bias = bias
        self.fea_dim = fea_dim

        self.projection1 = nn.Conv2d(fea_dim, fea_dim, kernel_size=1, padding=0, bias=self.bias)
        self.projection2 = nn.Conv2d(fea_dim, fea_dim, kernel_size=1, padding=0, bias=self.bias)
        self.filter1 = nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=self.bias, groups=fea_dim)
        self.filter2 = nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=self.bias, groups=fea_dim)

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
        y = F.conv2d(
            x[:, :c],
            self.filter1.weight[:c],
            self.filter1.bias[:c] if self.filter1.bias is not None else None,
            stride=self.filter1.stride,
            padding=self.filter1.padding,
            groups=c,
        )
        y = F.conv2d(
            y,
            self.projection1.weight[:c, :c],
            self.projection1.bias[:c] if self.projection1.bias is not None else None,
            stride=self.projection1.stride,
            padding=self.projection1.padding,
        )
        y = F.prelu(y, self.act.weight[:c])
        y = F.conv2d(
            y,
            self.filter2.weight[:c],
            self.filter2.bias[:c] if self.filter2.bias is not None else None,
            stride=self.filter2.stride,
            padding=self.filter2.padding,
            groups=c,
        )
        y = F.conv2d(
            y,
            self.projection2.weight[:c, :c],
            self.projection2.bias[:c] if self.projection2.bias is not None else None,
            stride=self.projection2.stride,
            padding=self.projection2.padding,
        )
        y = y * self.scale[:, :c] + x[:, :c]
        return y

    def param_num(self):
        
        total = 0
        total += sum(p.numel() for p in self.projection1.parameters())
        total += sum(p.numel() for p in self.filter1.parameters())
        total += sum(p.numel() for p in self.projection2.parameters())
        total += sum(p.numel() for p in self.filter2.parameters())

        return total

class DPSR(nn.Module):
    def __init__(self, scale, in_dim, fea_dim, num_blocks=5, bias=True):
        super(DPSR, self).__init__()

        self.scale = scale
        self.bias = bias
        self.fea_dim = fea_dim

        self.head = nn.Conv2d(in_dim, fea_dim, kernel_size=3, padding=1, bias=bias)

        self.body = nn.ModuleList()
        for _ in range(num_blocks):
            self.body.append(Block(fea_dim, bias=bias))

        self.tail = nn.Conv2d(fea_dim, in_dim * scale ** 2, kernel_size=3, padding=1, bias=bias)

        self.upsampler = nn.PixelShuffle(scale)
        self.alpha = nn.Parameter(torch.ones(1, 3, 1, 1))
        
    def forward(self, x):
        
        y = self.head(x)
        
        for i in range(len(self.body)):
            y = self.body[i](y)

        y = self.tail(y)
        y = self.alpha * self.upsampler(y)

        return y

    def forward_shared_channel(self, x, active_channels):
        c = max(1, min(int(active_channels), self.head.out_channels))
        y = F.conv2d(
            x,
            self.head.weight[:c],
            self.head.bias[:c] if self.head.bias is not None else None,
            stride=self.head.stride,
            padding=self.head.padding,
        )
        for i in range(len(self.body)):
            y = self.body[i].forward_shared_channel(y, c)

        y = F.conv2d(
            y,
            self.tail.weight[:,:c],
            self.tail.bias if self.tail.bias is not None else None,
            stride=self.tail.stride,
            padding=self.tail.padding
        )
        y = self.alpha * self.upsampler(y)
        return y
    
    def param_num(self):
        
        total = 0
        total += sum(p.numel() for p in self.head.parameters())

        for i in range(len(self.body)):
            total += self.body[i].param_num()

        total += sum(p.numel() for p in self.tail.parameters())

        return total

if __name__ == '__main__':
    
    model = DPSR(2, 3, 32, 5, bias=False)
    print(f"参数数量: {model.param_num()}")