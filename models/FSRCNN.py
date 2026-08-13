from torch import nn

class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        # Match the shared training interface without adding DPSR subnet behavior.
        self.scale = scale_factor
        self.fea_dim = d
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2, output_padding=scale_factor-1)

    def forward(self, x, channels=None):
        """Run the fixed-width FSRCNN path; ``channels`` is ignored for compatibility."""
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x

    def param_num(self):
        return sum(parameter.numel() for parameter in self.parameters())

# 测试FSRCNN网络
if __name__ == "__main__":
    model = FSRCNN(scale_factor=4)
    print(model)