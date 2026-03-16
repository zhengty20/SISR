import torch
import torch.nn as nn

class Block(nn.Module):
    
    def __init__(self, in_dim, out_dim, bias = True):
        super(Block, self).__init__()
        
        self.bias = bias
        self.mid_dim = in_dim

        self.projection1 = nn.Conv2d(in_dim, self.mid_dim, kernel_size=1, padding=0, bias=bias)
        self.filter1 = nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=3, padding=1, bias=bias, groups=in_dim)
        self.projection2 = nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=1, padding=0, bias=bias)
        self.filter2 = nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=3, padding=1, bias=bias, groups=in_dim)
        self.act = nn.ReLU(inplace=True)

        self.initialize_block_weights()

    def forward(self, x):

        y = self.projection1(x) 
        y = self.filter1(y)
        y = self.act(y)
        
        y = self.projection2(y)
        y = self.filter2(y)
        y = self.act(y) + x

        return y

    def param_num(self):
        
        total = 0
        total += sum(p.numel() for p in self.projection1.parameters())
        total += sum(p.numel() for p in self.filter1.parameters())
        total += sum(p.numel() for p in self.projection2.parameters())
        total += sum(p.numel() for p in self.filter2.parameters())

        return total
    
    def initialize_block_weights(self):

        nn.init.kaiming_normal_(self.projection1.weight, mode='fan_out', nonlinearity='linear')
        nn.init.kaiming_normal_(self.filter1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.projection2.weight, mode='fan_out', nonlinearity='linear')
        nn.init.kaiming_normal_(self.filter2.weight, mode='fan_out', nonlinearity='relu')

        if self.bias:
            nn.init.constant_(self.filter1.bias, 0)
            nn.init.constant_(self.filter2.bias, 0)
            nn.init.constant_(self.projection1.bias, 0)
            nn.init.constant_(self.projection2.bias, 0)

class ISCSR(nn.Module):

    def __init__(self, scale, in_dim, fea_dim, num_blocks=2, bias=True):
        super(ISCSR, self).__init__()

        self.scale = scale
        self.bias = bias

        self.head = nn.Conv2d(in_dim, fea_dim, kernel_size=3, padding=1, bias=bias)

        self.body = nn.ModuleList()
        for _ in range(num_blocks):
            self.body.append(Block(fea_dim, fea_dim, bias=bias))

        self.tail1 = nn.Conv2d(fea_dim, fea_dim, kernel_size=3, padding=1, bias=bias, groups=fea_dim)
        self.tail2 = nn.Conv2d(fea_dim, in_dim * scale ** 2, kernel_size=1, padding=0, bias=bias)

        self.upsampler = nn.PixelShuffle(scale)
        self.alpha = nn.Parameter(torch.zeros(1, 3, 1, 1))
        
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
    
    def initialize_weights(self):

        nn.init.kaiming_normal_(self.head.weight, mode='fan_out', nonlinearity='linear')
        nn.init.kaiming_normal_(self.tail1.weight, mode='fan_out', nonlinearity='linear')
        nn.init.kaiming_normal_(self.tail2.weight, mode='fan_out', nonlinearity='linear')


        if self.bias:
            nn.init.constant_(self.head1.bias, 0)
            nn.init.constant_(self.tail1.bias, 0)
            nn.init.constant_(self.tail2.bias, 0)

        
        nn.init.constant_(self.alpha, 0.1)

if __name__ == '__main__':
    
    model = ISCSR(2, 3, 36, 4, bias=False)
    print(f"参数数量: {model.param_num()}")