import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoundSTE(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class ScaleGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

class LSQPlusQuantizer(nn.Module):

    def __init__(self, bitwidth, is_symmetric=False, per_channel=False, is_activation=False):

        super().__init__()
        self.bitwidth = bitwidth
        self.is_symmetric = is_symmetric
        self.per_channel = per_channel
        self.is_activation = is_activation
        
        self.s = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

        # 定义量化范围 (n, p)
        if not self.is_symmetric:
            self.q_n = 0
            self.q_p = 2 ** self.bitwidth - 1
        else:
            self.q_n = - 2 ** (self.bitwidth - 1) + 1
            self.q_p = 2 ** (self.bitwidth - 1) - 1

        self.initialized = False

    def init_from_tensor(self, x):
        """
        根据输入张量 x 初始化量化参数 s 和 β。
        """
        if not self.is_activation:
            x_flat = x.detach().abs().view(x.size(0), -1)
            mean_val = x_flat.mean(dim=1)
            std_val = x_flat.std(dim=1)
            s_init = torch.max(torch.abs(mean_val - 3 * std_val), torch.abs(mean_val + 3 * std_val)) / 2 ** (self.bitwidth - 1)
            self.s.data = s_init.view(-1, 1, 1, 1)

        else: 
            x_min = x.detach().min()
            x_max = x.detach().max()
            self.s.data.fill_((x_max - x_min) / (self.q_p - self.q_n))
            self.beta.data.fill_(x_min - self.s.data * self.q_n)
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.init_from_tensor(x)

        # 1. 计算梯度缩放因子 g = 1 / sqrt(N * Q_p)，其中 N 是张量中元素的数量
        if self.per_channel:
             num_elements_per_channel = x.shape[1:].numel()
             grad_scale_factor = 1.0 / math.sqrt(num_elements_per_channel * self.q_p)
        else:
             grad_scale_factor = 1.0 / math.sqrt(x.numel() * self.q_p)

        # 2. 对 s 应用梯度缩放
        s_scaled = ScaleGradient.apply(self.s, grad_scale_factor)
        
        # 3. LSQ+ 量化: x_hat = round(clip((x - β) / s, n, p)) * s + β
        
        beta_scaled = self.beta
        x_normalized = (x - beta_scaled) / s_scaled
        x_clipped = torch.clamp(x_normalized, self.q_n, self.q_p)
        x_quantized_int = RoundSTE.apply(x_clipped)
        
        x_dequantized = x_quantized_int * s_scaled
        x_final = x_dequantized + beta_scaled
        
        return x_final

# -------------------------------------------------------------------
# 3. 封装层 (Wrapper Layer)
# -------------------------------------------------------------------

class QConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, weight_bitwidth=4, activation_bitwidth=4):
        
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding, 
                              groups=groups, bias=bias)
        
        self.weight_quantizer = LSQPlusQuantizer(bitwidth=weight_bitwidth, is_symmetric=True, per_channel=True, is_activation=False)
        self.activation_quantizer = LSQPlusQuantizer(bitwidth=activation_bitwidth, is_symmetric=False, per_channel=False, is_activation=True)

    def forward(self, x):

        quantized_x = self.activation_quantizer(x)
        quantized_weight = self.weight_quantizer(self.conv.weight)
        output = F.conv2d(quantized_x, quantized_weight, self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)
        
        return output