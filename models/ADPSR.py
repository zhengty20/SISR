import torch
import torch.nn as nn
import torch.nn.functional as F

from .DPSR import Block, USConv2d


class ResidualUpsamplingHead(nn.Module):
    """Generate an arbitrary-size HR residual from LR backbone features."""

    def __init__(
        self,
        fea_dim,
        out_dim,
        num_experts=4,
        controller_hidden=16,
        bias=False,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.experts = nn.ModuleList(
            USConv2d(
                fea_dim,
                out_dim,
                kernel_size=1,
                padding=0,
                bias=bias,
                us=(True, False),
            )
            for _ in range(self.num_experts)
        )
        self.controller = nn.Sequential(
            nn.Conv2d(4, controller_hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                controller_hidden,
                self.num_experts,
                kernel_size=1,
                bias=True,
            ),
        )

    @staticmethod
    def _coordinate_features(input_size, output_size, device, dtype):
        in_h, in_w = input_size
        out_h, out_w = output_size

        y = torch.arange(out_h, device=device, dtype=dtype)
        x = torch.arange(out_w, device=device, dtype=dtype)
        source_y = (y + 0.5) * (in_h / out_h) - 0.5
        source_x = (x + 0.5) * (in_w / out_w) - 0.5
        phase_y = source_y - torch.floor(source_y)
        phase_x = source_x - torch.floor(source_x)

        phase_y = phase_y.view(1, 1, out_h, 1).expand(1, 1, out_h, out_w)
        phase_x = phase_x.view(1, 1, 1, out_w).expand(1, 1, out_h, out_w)
        inv_scale_y = phase_y.new_full((1, 1, out_h, out_w), in_h / out_h)
        inv_scale_x = phase_x.new_full((1, 1, out_h, out_w), in_w / out_w)
        return torch.cat((phase_y, phase_x, inv_scale_y, inv_scale_x), dim=1)

    def forward(self, features, output_size):
        out_h, out_w = (int(output_size[0]), int(output_size[1]))
        in_h, in_w = features.shape[-2:]
        if out_h < in_h or out_w < in_w:
            raise ValueError(
                f"output_size must not be smaller than input features: "
                f"{(out_h, out_w)} vs {(in_h, in_w)}"
            )

        coordinate_features = self._coordinate_features(
            (in_h, in_w),
            (out_h, out_w),
            features.device,
            features.dtype,
        )
        routing = torch.softmax(self.controller(coordinate_features), dim=1)

        residual = None
        for expert_index, expert in enumerate(self.experts):
            expert_lr = expert(features)
            expert_hr = F.interpolate(
                expert_lr,
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            )
            weighted = expert_hr * routing[:, expert_index : expert_index + 1]
            residual = weighted if residual is None else residual + weighted
        return residual


class ADPSR(nn.Module):
    """DPSR backbone with a phase- and scale-aware arbitrary residual head."""

    def __init__(
        self,
        in_dim,
        fea_dim,
        num_blocks=5,
        bias=False,
        subnet_channels=16,
        num_experts=4,
        controller_hidden=16,
    ):
        super().__init__()
        if not 0 < subnet_channels < fea_dim:
            raise ValueError(
                f"subnet_channels must be in [1, {fea_dim - 1}], "
                f"got {subnet_channels}"
            )
        self.fea_dim = int(fea_dim)
        self.subnet_channels = int(subnet_channels)
        self.num_blocks = int(num_blocks)
        self.head = USConv2d(
            in_dim,
            fea_dim,
            kernel_size=1,
            padding=0,
            bias=bias,
            us=(False, True),
        )
        self.body = nn.ModuleList(
            Block(fea_dim, bias=bias) for _ in range(num_blocks)
        )
        self.residual_head = ResidualUpsamplingHead(
            fea_dim=fea_dim,
            out_dim=in_dim,
            num_experts=num_experts,
            controller_hidden=controller_hidden,
            bias=bias,
        )

    @property
    def supported_channels(self):
        return (self.fea_dim, self.subnet_channels)

    def forward(self, x, output_size, channels=None):
        channels = self.fea_dim if channels is None else int(channels)
        if channels not in self.supported_channels:
            raise ValueError(
                f"channels must be one of {self.supported_channels}, got {channels}"
            )
        y = self.head(x, active_out_channels=channels)
        for block in self.body:
            y = block(y)
        return self.residual_head(y, output_size)

    def param_num(self):
        return sum(parameter.numel() for parameter in self.parameters())
