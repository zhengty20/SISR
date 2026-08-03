import torch
import torch.nn.functional as F


def rgb_to_gray(image):
    """Convert an NCHW [0, 255] RGB image to conventional grayscale values."""
    if image.ndim != 4:
        raise ValueError(f"Expected NCHW image, got shape {tuple(image.shape)}")
    if image.shape[1] == 1:
        return image
    if image.shape[1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {image.shape[1]}")
    coefficients = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (image * coefficients).sum(dim=1, keepdim=True)


def rgb_to_studio_y(image):
    """Convert an NCHW [0, 255] RGB image to the studio-range Y channel."""
    if image.ndim != 4:
        raise ValueError(f"Expected NCHW image, got shape {tuple(image.shape)}")
    if image.shape[1] == 1:
        return image
    if image.shape[1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {image.shape[1]}")
    coefficients = image.new_tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1)
    return (image * coefficients).sum(dim=1, keepdim=True) / 255.0 + 16.0


def laplacian_map(image_y):
    """Absolute eight-neighbor Laplace response with replicate boundaries."""
    if image_y.ndim != 4 or image_y.shape[1] != 1:
        raise ValueError(f"Expected N1HW image, got shape {tuple(image_y.shape)}")
    kernel = 0.125 * image_y.new_tensor(
        [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]
    ).view(1, 1, 3, 3)
    padded = F.pad(image_y, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(padded, kernel).abs()
