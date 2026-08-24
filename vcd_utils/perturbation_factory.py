from PIL import Image

from vcd_utils.skin_perturb import add_skin_perturbation
from vcd_utils.vcd_add_noise import add_diffusion_noise


def build_contrast_image(processor, image_path: str, image_tensor, method: str, noise_step: int, strength: float):
    method = method.lower()
    if method == "vcd":
        return add_diffusion_noise(image_tensor, noise_step)

    image = Image.open(image_path).convert("RGB")
    perturbed = add_skin_perturbation(image, method=method, strength=strength)
    image_processor = getattr(processor, "image_processor", processor)
    processed = image_processor(images=perturbed, return_tensors="pt")
    return processed["pixel_values"].to(device=image_tensor.device, dtype=image_tensor.dtype)