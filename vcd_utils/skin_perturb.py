import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image


def _minmax_image(image_tensor: torch.Tensor) -> torch.Tensor:
    mins = image_tensor.amin(dim=(-2, -1), keepdim=True)
    maxs = image_tensor.amax(dim=(-2, -1), keepdim=True)
    return (image_tensor - mins) / (maxs - mins + 1e-6)


def _estimate_lesion_mask(image_tensor: torch.Tensor) -> torch.Tensor:
    image_01 = _minmax_image(image_tensor)
    border = torch.zeros_like(image_01[:, :1])
    border[:, :, 0, :] = 1
    border[:, :, -1, :] = 1
    border[:, :, :, 0] = 1
    border[:, :, :, -1] = 1

    border_pixels = border.expand_as(image_01).bool()
    border_values = image_01.masked_select(border_pixels).view(image_01.shape[0], image_01.shape[1], -1)
    border_mean = border_values.mean(dim=-1, keepdim=True).unsqueeze(-1)
    contrast_score = (image_01 - border_mean).abs().mean(dim=1, keepdim=True)

    height, width = image_01.shape[-2:]
    ys = torch.linspace(-1.0, 1.0, height, device=image_01.device).view(1, 1, height, 1)
    xs = torch.linspace(-1.0, 1.0, width, device=image_01.device).view(1, 1, 1, width)
    center_prior = torch.exp(-3.5 * (xs.square() + ys.square()))

    lesion_score = contrast_score * (0.65 + 0.35 * center_prior)
    flat = lesion_score.flatten(start_dim=2)
    threshold = torch.quantile(flat, 0.65, dim=-1, keepdim=True).view(-1, 1, 1, 1)
    mask = (lesion_score >= threshold).float()
    kernel = 15 if min(height, width) >= 64 else 7
    mask = TF.gaussian_blur(mask, kernel_size=kernel)
    return mask.clamp_(0.0, 1.0)


def _border_ring(mask: torch.Tensor, radius: int = 9) -> torch.Tensor:
    dilated = F.max_pool2d(mask, kernel_size=radius, stride=1, padding=radius // 2)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=radius, stride=1, padding=radius // 2)
    return (dilated - eroded).clamp_(0.0, 1.0)


def _desaturate_region(image_tensor: torch.Tensor, mask: torch.Tensor, strength: float = 0.75) -> torch.Tensor:
    image_01 = _minmax_image(image_tensor)
    gray = TF.rgb_to_grayscale(image_01, num_output_channels=image_01.shape[1])
    target = image_01.lerp(gray, strength)
    mixed = image_01 * (1.0 - mask) + target * mask
    return image_tensor + (mixed - image_01)


def _desaturate_global(image_tensor: torch.Tensor, strength: float = 0.2) -> torch.Tensor:
    image_01 = _minmax_image(image_tensor)
    gray = TF.rgb_to_grayscale(image_01, num_output_channels=image_01.shape[1])
    mixed = image_01.lerp(gray, strength)
    return image_tensor + (mixed - image_01)


def _blur_region(image_tensor: torch.Tensor, mask: torch.Tensor, kernel_size: int = 21, strength: float = 1.0) -> torch.Tensor:
    blurred = TF.gaussian_blur(image_tensor, kernel_size=kernel_size)
    blend = (mask * strength).clamp_(0.0, 1.0)
    return image_tensor * (1.0 - blend) + blurred * blend


def _occlude_region(image_tensor: torch.Tensor, mask: torch.Tensor, fill_mode: str = "border") -> torch.Tensor:
    if fill_mode == "border":
        border = torch.zeros_like(mask)
        border[:, :, 0, :] = 1
        border[:, :, -1, :] = 1
        border[:, :, :, 0] = 1
        border[:, :, :, -1] = 1
        border_values = image_tensor.masked_select(border.expand_as(image_tensor).bool()).view(image_tensor.shape[0], image_tensor.shape[1], -1)
        fill = border_values.mean(dim=-1, keepdim=True).unsqueeze(-1)
    else:
        fill = image_tensor.mean(dim=(-2, -1), keepdim=True)
    return image_tensor * (1.0 - mask) + fill * mask


def _pil_to_bchw(image: Image.Image) -> torch.Tensor:
    return TF.to_tensor(image).unsqueeze(0)


def _bchw_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    return TF.to_pil_image(image_tensor.squeeze(0).clamp(0.0, 1.0))


def add_skin_perturbation(image: Image.Image, method: str, strength: float = 1.0) -> Image.Image:
    image_tensor = _pil_to_bchw(image)
    mask = _estimate_lesion_mask(image_tensor)

    if method == "color":
        ring = _border_ring(mask, radius=13)
        lesion_strength = min(1.0, 0.7 + 0.08 * strength)
        ring_strength = min(0.9, 0.35 + 0.05 * strength)
        global_strength = min(0.45, 0.08 + 0.03 * strength)

        output = _desaturate_global(image_tensor, strength=global_strength)
        output = _desaturate_region(output, ring, strength=ring_strength)
        output = _desaturate_region(output, mask, strength=lesion_strength)
        return _bchw_to_pil(output)

    if method == "boundary_blur":
        ring = _border_ring(mask, radius=9)
        output = _blur_region(image_tensor, ring, kernel_size=17, strength=min(1.0, 0.6 + 0.1 * strength))
        return _bchw_to_pil(output)

    if method == "texture_blur":
        output = _blur_region(image_tensor, mask, kernel_size=21, strength=min(1.0, 0.45 + 0.08 * strength))
        return _bchw_to_pil(output)

    if method == "occlusion":
        core_mask = (mask > 0.6).float()
        output = _occlude_region(image_tensor, core_mask)
        return _bchw_to_pil(output)

    if method == "skin_morph":
        output = _desaturate_region(image_tensor, mask, strength=0.7)
        output = _blur_region(output, _border_ring(mask, radius=9), kernel_size=17, strength=0.7)
        output = _blur_region(output, mask, kernel_size=21, strength=0.45)
        core_mask = (mask > 0.7).float() * 0.4
        output = _occlude_region(output, core_mask)
        return _bchw_to_pil(output)

    raise ValueError(f"Unsupported skin perturbation method: {method}")