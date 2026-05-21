import torch
import torch.nn.functional as F
        
def bilinear_interpolate_hdl(x: torch.Tensor, scale: float) -> torch.Tensor:
    """
    8bit 定点 bilinear 向量化实现
    """
    if x.ndim != 4:
        raise ValueError("x must be 4D tensor with shape (B, C, H, W)")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    scale = int(scale)
    _, _, in_h, in_w = x.shape
    out_h = int(in_h * scale)
    out_w = int(in_w * scale)
    device = x.device

    x = x.round().clamp(0, 255).to(torch.int32)

    y = torch.arange(out_h, device=device, dtype=torch.int64)
    x_idx = torch.arange(out_w, device=device, dtype=torch.int64)

    table_map = {
        2: torch.tensor([3, 1], device=device, dtype=torch.int32),
        3: torch.tensor([4, 0, 2], device=device, dtype=torch.int32),
        4: torch.tensor([5, 7, 1, 3], device=device, dtype=torch.int32),
    }

    denom = 2 * scale
    y0 = (2 * y + 1 - scale) // (2 * scale)
    x0 = (2 * x_idx + 1 - scale) // (2 * scale)
    y1 = y0 + 1
    x1 = x0 + 1
    y0c = y0.clamp(0, in_h - 1)
    x0c = x0.clamp(0, in_w - 1)
    y1c = y1.clamp(0, in_h - 1)
    x1c = x1.clamp(0, in_w - 1)
    
    wy1 = table_map[scale][y % scale]
    wx1 = table_map[scale][x_idx % scale]
    wy0 = denom - wy1
    wx0 = denom - wx1
    wx0 = wx0.view(1, 1, 1, out_w)
    wx1 = wx1.view(1, 1, 1, out_w)
    wy0 = wy0.view(1, 1, out_h, 1)
    wy1 = wy1.view(1, 1, out_h, 1)

    p00 = x[:, :, y0c[:, None], x0c[None, :]]
    p01 = x[:, :, y0c[:, None], x1c[None, :]]
    p10 = x[:, :, y1c[:, None], x0c[None, :]]
    p11 = x[:, :, y1c[:, None], x1c[None, :]]
    
    if scale == 2 or scale == 4:
        out = ((p00 * wx0 + p01 * wx1) / denom).floor() * wy0 + ((p10 * wx0 + p11 * wx1) / denom).floor() * wy1
        out = (out / denom).floor().to(torch.float32)
    else:
        out1 = ((p00 * wx0 + p01 * wx1) / 8).floor() + ((p00 * wx0 + p01 * wx1) / 32).floor() + \
        ((p00 * wx0 + p01 * wx1) / 128).floor() + ((p00 * wx0 + p01 * wx1) / 256).floor()
        out2 = ((p10 * wx0 + p11 * wx1) / 8).floor() + ((p10 * wx0 + p11 * wx1) / 32).floor() + \
        ((p10 * wx0 + p11 * wx1) / 128).floor() + ((p10 * wx0 + p11 * wx1) / 256).floor()
        out = out1 * wy0 + out2 * wy1
        out = (out / 8).floor() + (out / 32).floor() + (out / 128).floor() + (out / 256).floor()
    
    return out.to(torch.float32)

def bilinear_interpolation(x: torch.Tensor, scale: float, bit8: bool = False) -> torch.Tensor:
    if bit8:
        return bilinear_interpolate_hdl(x, scale).floor()
    return F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False).floor()
