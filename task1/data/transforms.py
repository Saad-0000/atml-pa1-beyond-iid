import torch
from task1.configs import configs
import torchvision.transforms.v2.functional as TF


def apply_grayscale(imgs: torch.Tensor) -> torch.Tensor:
    """strips chromatic information; preserves luminance."""
    return TF.rgb_to_grayscale(imgs, num_output_channels=3)


def apply_hue_rotation(
    imgs: torch.Tensor, hue_factor: float = 0.5
) -> torch.Tensor:
    """180-degree hue inversion (factor 0.5 maps to 180 degrees in torchvision)."""
    return TF.adjust_hue(imgs, hue_factor=hue_factor)


def apply_cardinal_translation(
    imgs: torch.Tensor, delta: int, direction: str
) -> torch.Tensor:
    """Shifts by delta pixels in cardinal directions using reflection padding."""
    if delta == 0:
        return imgs

    pad_dict = {
        "up": (0, 0, 0, delta),
        "down": (0, delta, 0, 0),
        "left": (0, 0, delta, 0),
        "right": (delta, 0, 0, 0),
    }
    padded = TF.pad(imgs, pad_dict[direction], padding_mode="reflect")
    _, _, h, w = imgs.shape

    if direction == "up":
        return padded[:, :, delta : delta + h, :w]
    elif direction == "down":
        return padded[:, :, 0:h, :w]
    elif direction == "left":
        return padded[:, :, :h, delta : delta + w]
    elif direction == "right":
        return padded[:, :, :h, 0:w]
    raise ValueError(f"Unknown direction {direction}")


def apply_patch_shuffle(
    imgs: torch.Tensor, grid_size: int = 4, seed: int = configs.SEED
) -> torch.Tensor:
    """Partitions image into grid_size x grid_size blocks and applies fixed permutation."""
    b, c, h, w = imgs.shape
    ph, pw = h // grid_size, w // grid_size
    num_patches = grid_size * grid_size

    # Split into patches: (B, C, grid_size, ph, grid_size, pw)
    patches = (
        imgs.unfold(2, ph, ph)
        .unfold(3, pw, pw)
        .permute(0, 2, 4, 1, 3, 5)
        .contiguous()
    )
    patches = patches.view(b, num_patches, c, ph, pw)

    # Generate reproducible non-identity permutation
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_patches, generator=g)
    while torch.equal(perm, torch.arange(num_patches)):
        perm = torch.randperm(num_patches, generator=g)

    shuffled_patches = patches[:, perm, :, :, :]

    # Reconstruct image
    shuffled = shuffled_patches.view(b, grid_size, grid_size, c, ph, pw)
    shuffled = (
        shuffled.permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .view(b, c, grid_size * ph, grid_size * pw)
    )
    return shuffled