import torch
import torch.nn.functional as F
from models.Bilinear import bilinear_interpolate_hdl

def interp(image: torch.Tensor, scale_factor: int):
    """torch版定点双线性插值（单通道2D输入）。"""
    if image.ndim != 2:
        raise ValueError("image must be 2D tensor")
    if scale_factor <= 0:
        raise ValueError("scale_factor must be > 0")
    
    image = image.to(torch.int32)
    
    table_map = {
        2: torch.tensor([3, 1], device=image.device, dtype=torch.int32),
        3: torch.tensor([4, 0, 2], device=image.device, dtype=torch.int32),
        4: torch.tensor([5, 7, 1, 3], device=image.device, dtype=torch.int32),
    }

    u_lut = table_map[scale_factor]
    v_lut = u_lut
    h_in, w_in = image.shape
    h_out, w_out = h_in * scale_factor, w_in * scale_factor

    y = torch.arange(h_out, device=image.device, dtype=torch.int64)
    x = torch.arange(w_out, device=image.device, dtype=torch.int64)
    y0 = (2 * y + 1 - scale_factor) // (2 * scale_factor)
    x0 = (2 * x + 1 - scale_factor) // (2 * scale_factor)
    y1 = y0 + 1
    x1 = x0 + 1

    y0c = y0.clamp(0, h_in - 1)
    y1c = y1.clamp(0, h_in - 1)
    x0c = x0.clamp(0, w_in - 1)
    x1c = x1.clamp(0, w_in - 1)

    u = u_lut[x % scale_factor].view(1, w_out)
    v = v_lut[y % scale_factor].view(h_out, 1)
    denom = 2 * scale_factor

    p11 = image[y0c[:, None], x0c[None, :]]
    p21 = image[y0c[:, None], x1c[None, :]]
    p12 = image[y1c[:, None], x0c[None, :]]
    p22 = image[y1c[:, None], x1c[None, :]]

    top = ((denom - u) * p11 + u * p21) // denom
    bot = ((denom - u) * p12 + u * p22) // denom
    out = ((denom - v) * top + v * bot) // denom
    return out.clamp(0, 255)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_image = torch.tensor([[60, 70, 80, 90], [70, 80, 90, 100]], dtype=torch.float32, device=device)
    scale = 2

    dst_image1 = interp(src_image, scale)
    print(dst_image1.to(torch.float32))

    src_4d = src_image.view(1, 1, src_image.shape[0], src_image.shape[1])
    dst_image2 = bilinear_interpolate_hdl(src_4d, scale)
    # print(dst_image2.view(dst_image2.shape[0], dst_image2.shape[1]).to(torch.float32))
    print((dst_image1.to(torch.float32) - dst_image2.to(torch.float32)).abs().mean().item())

    dst_image_fp32 = F.interpolate(src_4d, scale_factor=scale, mode="bilinear", align_corners=False).floor().squeeze(0).squeeze(0)
    # print(dst_image_fp32)
    print((dst_image2.to(torch.float32) - dst_image_fp32).abs().mean().item())

if __name__ == "__main__":
    main()