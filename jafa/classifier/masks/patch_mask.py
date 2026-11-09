import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .mask_function import MaskFunction


class PatchMask(MaskFunction):
    """
    Randomly masks patches from an image.
    
    Args:
        patch_size (int): Size of each patch (e.g., 8 for 8x8 patches)
        mask_ratio (float): Fraction of patches to mask (0.0 to 1.0)
        mask_value (float): Value to use for masked patches (default: 0.0)
        random_seed (Optional[int]): Random seed for reproducibility
    """
    
    def __init__(
        self, 
        patch_size: int = 8, 
        mask_ratio: float = 0.5, 
        mask_value: float = 0.0,
        random_seed: Optional[int] = None,
        device: torch.device = torch.device("cpu") # shaan : why is this set to cpu?
    ):
        # For PatchMask, we don't need the full MaskFunction parameters
        # We'll create a minimal target_shape for compatibility
        target_shape = (32, 32)  # Default CIFAR-10 size
        k_fraction = mask_ratio  # Use mask_ratio as k_fraction for compatibility
        batch_size = 1  # Will be overridden in forward
        sampled_indices = (0, -1)
        center_fraction = 0.0
        in_channels = 3
        is_complex = False
        super().__init__(
            target_shape=target_shape,
            k_fraction=k_fraction,
            batch_size=batch_size,
            sampled_indices=sampled_indices,
            center_fraction=center_fraction,
            in_channels=in_channels,
            is_complex=is_complex,
            device=device,
        )
        
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
    
    def forward(self, x: torch.Tensor, batch_size: int = None, return_mask: bool = False):
        """
        Apply patch masking to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            batch_size: Ignored for patch masking (kept for interface compatibility)
            return_mask: If True, return both masked tensor and binary mask
            
        Returns:
            torch.Tensor or tuple: If return_mask=False, returns masked tensor.
                                  If return_mask=True, returns (masked_tensor, binary_mask)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {x.dim()}D")
        
        B, C, H, W = x.shape
        
        # Calculate number of patches
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        total_patches = num_patches_h * num_patches_w
        
        # Calculate number of patches to mask
        num_patches_to_mask = int(total_patches * self.mask_ratio)
        
        # Create mask tensor
        mask = torch.ones(B, total_patches, device=x.device)
        
        # Randomly select patches to mask
        for b in range(B):
            patch_indices = torch.randperm(total_patches, device=x.device)[:num_patches_to_mask]
            mask[b, patch_indices] = 0

        # shaan : line below
        flattened_mask_indices = mask

        # Reshape mask to match image dimensions
        mask = mask.view(B, 1, num_patches_h, num_patches_w)
        mask = F.interpolate(
            mask, 
            size=(H, W), 
            mode='nearest'
        )
        
        # Apply mask
        masked_x = x * mask + self.mask_value * (1 - mask)
        
        #shaan
        return_mask = True
        if return_mask:
            # return masked_x, mask
            # shaan
            return masked_x, flattened_mask_indices, mask
        else:
            return masked_x
    

    # Shaan : commented out below function
    # def get_mask(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Get the binary mask without applying it.
        
    #     Args:
    #         x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
    #     Returns:
    #         torch.Tensor: Binary mask tensor of shape (B, 1, H, W)
    #     """
    #     if x.dim() != 4:
    #         raise ValueError(f"Expected 4D tensor (B, C, H, W), got {x.dim()}D")
        
    #     B, C, H, W = x.shape
        
    #     # Calculate number of patches
    #     num_patches_h = H // self.patch_size
    #     num_patches_w = W // self.patch_size
    #     total_patches = num_patches_h * num_patches_w
        
    #     # Calculate number of patches to mask
    #     num_patches_to_mask = int(total_patches * self.mask_ratio)
        
    #     # Create mask tensor
    #     mask = torch.ones(B, total_patches, device=x.device)
        
    #     # Randomly select patches to mask
    #     for b in range(B):
    #         patch_indices = torch.randperm(total_patches, device=x.device)[:num_patches_to_mask]
    #         mask[b, patch_indices] = 0
        
    #     # Reshape mask to match image dimensions
    #     mask = mask.view(B, 1, num_patches_h, num_patches_w)
    #     mask = F.interpolate(
    #         mask, 
    #         size=(H, W), 
    #         mode='nearest'
    #     )
        
    #     return mask


# Shaan commented out below function

# def apply_patch_mask(
#     image: torch.Tensor, 
#     patch_size: int = 8, 
#     mask_ratio: float = 0.5, 
#     mask_value: float = 0.0
# ) -> torch.Tensor:
#     """
#     Convenience function to apply patch masking to an image.
    
#     Args:
#         image (torch.Tensor): Input image tensor of shape (B, C, H, W)
#         patch_size (int): Size of each patch
#         mask_ratio (float): Fraction of patches to mask
#         mask_value (float): Value to use for masked patches
        
#     Returns:
#         torch.Tensor: Masked image tensor
#     """
#     mask_fn = PatchMask(patch_size, mask_ratio, mask_value)
#     return mask_fn(image) 