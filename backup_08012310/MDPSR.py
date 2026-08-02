import torch.nn as nn

# Block with compressed channel
class BlockC(nn.Module):
    def __init__(self, fea_dim, bias = True):
        super(BlockC, self).__init__() 
        self.bias = bias
        self.fea_dim = fea_dim

        self.projection1 = nn.Conv2d(fea_dim // 2, fea_dim // 2, kernel_size=1, padding=0, bias=self.bias, padding_mode='replicate')
        self.projection2 = nn.Conv2d(fea_dim // 2, fea_dim // 2, kernel_size=1, padding=0, bias=self.bias, padding_mode='replicate')
        self.filter1 = nn.Conv2d(fea_dim // 2, fea_dim // 2, kernel_size=3, padding=1, bias=self.bias, groups=fea_dim // 2, padding_mode='replicate')
        self.filter2 = nn.Conv2d(fea_dim // 2, fea_dim // 2, kernel_size=3, padding=1, bias=self.bias, groups=fea_dim // 2, padding_mode='replicate')

        self.act1 = nn.PReLU(num_parameters=self.fea_dim // 2, init=0.25)
        self.act2 = nn.PReLU(num_parameters=self.fea_dim // 2, init=0.25)

    def forward(self, x):
        y = self.filter1(x)
        y = self.projection1(y)
        y = self.act1(y)
        
        y = self.filter2(y)
        y = self.projection2(y)
        y = self.act2(y)

        return y

    def param_num(self):      
        total = 0
        total += sum(p.numel() for p in self.projection1.parameters())
        total += sum(p.numel() for p in self.filter1.parameters())
        total += sum(p.numel() for p in self.projection2.parameters())
        total += sum(p.numel() for p in self.filter2.parameters())

        return total

# Bridge block
class BlockB(nn.Module):
    def __init__(self, fea_dim, bias = True):
        super(BlockB, self).__init__() 
        self.bias = bias
        self.fea_dim = fea_dim

        self.projection1 = nn.Conv2d(fea_dim // 2, fea_dim // 2, kernel_size=1, padding=0, bias=self.bias, padding_mode='replicate')
        self.projection2 = nn.Conv2d(fea_dim // 2, fea_dim, kernel_size=1, padding=0, bias=self.bias, padding_mode='replicate')
        self.filter1 = nn.Conv2d(fea_dim // 2, fea_dim // 2, kernel_size=3, padding=1, bias=self.bias, groups=fea_dim // 2, padding_mode='replicate')
        self.filter2 = nn.Conv2d(fea_dim // 2, fea_dim // 2, kernel_size=3, padding=1, bias=self.bias, groups=fea_dim // 2, padding_mode='replicate')

        self.act1 = nn.PReLU(num_parameters=self.fea_dim // 2, init=0.25)
        self.act2 = nn.PReLU(num_parameters=self.fea_dim, init=0.25)

    def forward(self, x):
        y = self.filter1(x)
        y = self.projection1(y)
        y = self.act1(y)
        
        y = self.filter2(y)
        y = self.projection2(y)
        y = self.act2(y)

        return y

    def param_num(self):      
        total = 0
        total += sum(p.numel() for p in self.projection1.parameters())
        total += sum(p.numel() for p in self.filter1.parameters())
        total += sum(p.numel() for p in self.projection2.parameters())
        total += sum(p.numel() for p in self.filter2.parameters())

        return total

class Block(nn.Module):
    def __init__(self, fea_dim, bias = True):
        super(Block, self).__init__() 
        self.bias = bias
        self.fea_dim = fea_dim

        self.projection1 = nn.Conv2d(fea_dim, fea_dim, kernel_size=1, padding=0, bias=self.bias, padding_mode='replicate')
        self.projection2 = nn.Conv2d(fea_dim, fea_dim, kernel_size=1, padding=0, bias=self.bias, padding_mode='replicate')
        self.filter1 = nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=self.bias, groups=fea_dim, padding_mode='replicate')
        self.filter2 = nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=self.bias, groups=fea_dim, padding_mode='replicate')

        self.act1 = nn.PReLU(num_parameters=self.fea_dim, init=0.25)
        self.act2 = nn.PReLU(num_parameters=self.fea_dim, init=0.25)

    def forward(self, x):
        y = self.filter1(x)
        y = self.projection1(y)
        y = self.act1(y)
        
        y = self.filter2(y)
        y = self.projection2(y)
        y = self.act2(y)

        return y

    def param_num(self):      
        total = 0
        total += sum(p.numel() for p in self.projection1.parameters())
        total += sum(p.numel() for p in self.filter1.parameters())
        total += sum(p.numel() for p in self.projection2.parameters())
        total += sum(p.numel() for p in self.filter2.parameters())

        return total

class MDPSR(nn.Module):
    def __init__(self, scale, in_dim, fea_dim, num_blocks=5, mixed_blocks=3, bias=True):
        super(MDPSR, self).__init__()
        if fea_dim < 2 or fea_dim % 2:
            raise ValueError("MDPSR requires a positive even fea_dim")
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if not 1 <= mixed_blocks <= num_blocks:
            raise ValueError("mixed_blocks must be in [1, num_blocks]")

        self.scale = scale
        self.bias = bias
        self.fea_dim = fea_dim
        self.num_blocks = int(num_blocks)
        self.mixed_blocks = int(mixed_blocks)
        self.head = nn.Conv2d(
            in_dim, fea_dim // 2, kernel_size=1, padding=0, bias=bias
        )

        self.body = nn.ModuleList()
        for i in range(self.num_blocks):
            if i < self.mixed_blocks - 1:
                self.body.append(BlockC(fea_dim, bias=bias))
            elif i == self.mixed_blocks - 1:
                self.body.append(BlockB(fea_dim, bias=bias))
            else:
                self.body.append(Block(fea_dim, bias=bias))

        self.tail = nn.Conv2d(
            fea_dim, in_dim * scale ** 2, kernel_size=1, padding=0, bias=bias
        )
        self.upsampler = nn.PixelShuffle(scale)
        
    def forward(self, x):
        
        y = self.head(x)
        
        for i in range(len(self.body)):
            y = self.body[i](y)

        y = self.tail(y)
        y = self.upsampler(y)

        return y
    
    def param_num(self):
        
        total = 0
        total += sum(p.numel() for p in self.head.parameters()) 

        for i in range(len(self.body)):
            total += self.body[i].param_num()

        total += sum(p.numel() for p in self.tail.parameters())
        
        return total

if __name__ == '__main__':
    
    model = MDPSR(2, 3, 32, 5, bias=False)
    print(f"参数数量: {model.param_num()}")
