"""
Realistic Shadow Generator - Core Logic
Generates physically-inspired shadows with:
- Directional light control (angle + elevation)
- Contact shadows (sharp near contact, fading with distance)
- Soft shadow falloff (blur increases with distance)
- Depth map warping (bonus)
"""

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from typing import Tuple, Optional
import math


class ShadowGenerator:
    """Generate realistic shadows for compositing foreground onto background."""

    def __init__(
        self,
        light_angle: float = 135.0,      # 0-360 degrees (direction light comes FROM)
        light_elevation: float = 45.0,    # 0-90 degrees (height of light source)
        shadow_intensity: float = 0.7,    # 0-1 (darkness of shadow)
        contact_shadow_size: int = 15,    # pixels for contact shadow
        max_blur: float = 30.0,           # maximum blur at shadow tip
        shadow_length_factor: float = 1.5 # how long shadow extends
    ):
        self.light_angle = light_angle
        self.light_elevation = light_elevation
        self.shadow_intensity = shadow_intensity
        self.contact_shadow_size = contact_shadow_size
        self.max_blur = max_blur
        self.shadow_length_factor = shadow_length_factor

    def extract_alpha_mask(self, foreground: Image.Image) -> np.ndarray:
        """Extract alpha channel as mask (0-1 float array)."""
        if foreground.mode != 'RGBA':
            foreground = foreground.convert('RGBA')

        alpha = np.array(foreground.split()[-1], dtype=np.float32) / 255.0
        return alpha

    def compute_shadow_offset(self, height: int) -> Tuple[float, float]:
        """
        Compute shadow offset based on light angle and elevation.
        Returns (dx, dy) offset in pixels.
        """
        # Convert angles to radians
        angle_rad = math.radians(self.light_angle)
        elevation_rad = math.radians(self.light_elevation)

        # Shadow length inversely related to elevation (lower sun = longer shadow)
        # At 90° elevation (directly overhead), shadow length approaches 0
        # At 0° elevation (horizon), shadow is very long
        shadow_length = height * self.shadow_length_factor * math.tan(math.radians(90 - self.light_elevation))
        shadow_length = min(shadow_length, height * 3)  # Cap at 3x height

        # Shadow direction is opposite to light direction
        dx = shadow_length * math.cos(angle_rad + math.pi)
        dy = shadow_length * math.sin(angle_rad + math.pi)

        return dx, dy

    def create_distance_field(self, mask: np.ndarray) -> np.ndarray:
        """
        Create a distance field from the mask.
        Values represent distance from the nearest opaque pixel.
        Used for graduated blur and opacity falloff.
        """
        # Binary mask (threshold at 0.5)
        binary = mask > 0.5

        # Distance transform - distance from each pixel to nearest foreground pixel
        distance = ndimage.distance_transform_edt(~binary)

        return distance

    def create_contact_shadow(self, mask: np.ndarray) -> np.ndarray:
        """
        Create a sharp contact shadow near the base of the object.
        This simulates ambient occlusion at contact points.
        """
        # Find the bottom edge of the object (contact area)
        binary = mask > 0.1

        # Dilate slightly for contact region
        contact = ndimage.binary_dilation(binary, iterations=self.contact_shadow_size // 2)
        contact = contact.astype(np.float32)

        # Apply slight blur for softness
        contact_img = Image.fromarray((contact * 255).astype(np.uint8))
        contact_img = contact_img.filter(ImageFilter.GaussianBlur(radius=3))
        contact = np.array(contact_img, dtype=np.float32) / 255.0

        # Mask out the foreground object itself
        contact = contact * (1 - mask)

        return contact * 0.8  # Contact shadow intensity

    def create_cast_shadow(
        self,
        mask: np.ndarray,
        canvas_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create the main cast shadow with perspective projection.
        Returns (shadow_mask, blur_map) for variable blur application.
        """
        h, w = mask.shape
        canvas_h, canvas_w = canvas_size

        # Get shadow offset
        dx, dy = self.compute_shadow_offset(h)

        # Create output arrays
        shadow = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        blur_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        # Find object bounds for processing
        rows = np.any(mask > 0.1, axis=1)
        cols = np.any(mask > 0.1, axis=0)

        if not rows.any() or not cols.any():
            return shadow, blur_map

        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        object_height = row_max - row_min
        object_base = row_max  # Bottom of object

        # Project shadow with perspective (shadow stretches from base)
        for y in range(row_min, row_max + 1):
            # Distance from base (contact point)
            dist_from_base = object_base - y

            # Shadow offset increases with height (perspective)
            # Points higher up cast shadows further away
            progress = dist_from_base / max(object_height, 1)

            # Current row's shadow offset
            row_dx = dx * progress
            row_dy = dy * progress

            for x in range(col_min, col_max + 1):
                if mask[y, x] > 0.1:
                    # Project this pixel to shadow position
                    shadow_x = int(x + row_dx)
                    shadow_y = int(y + row_dy + dist_from_base * 0.3)  # Shadow falls down

                    # Bounds check
                    if 0 <= shadow_x < canvas_w and 0 <= shadow_y < canvas_h:
                        # Shadow intensity decreases with distance from base
                        intensity = mask[y, x] * (1 - 0.5 * progress)
                        shadow[shadow_y, shadow_x] = max(shadow[shadow_y, shadow_x], intensity)

                        # Blur increases with distance from contact
                        blur_map[shadow_y, shadow_x] = progress

        return shadow, blur_map

    def apply_variable_blur(
        self,
        shadow: np.ndarray,
        blur_map: np.ndarray
    ) -> np.ndarray:
        """
        Apply variable Gaussian blur based on the blur map.
        Areas with higher blur_map values get more blur (farther from contact).
        """
        # Create multiple blur levels
        blur_levels = 8
        result = np.zeros_like(shadow)

        for i in range(blur_levels):
            progress = i / (blur_levels - 1)
            blur_radius = 1 + progress * self.max_blur

            # Create blurred version
            shadow_img = Image.fromarray((shadow * 255).astype(np.uint8))
            blurred = shadow_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            blurred_arr = np.array(blurred, dtype=np.float32) / 255.0

            # Blend based on blur map
            # Pixels with blur_map close to progress value use this blur level
            weight_range = 1.0 / blur_levels
            lower = max(0, progress - weight_range)
            upper = min(1, progress + weight_range)

            weight = np.clip((blur_map - lower) / (upper - lower + 0.001), 0, 1)
            if i < blur_levels - 1:
                next_lower = (i + 1) / (blur_levels - 1) - weight_range
                weight *= np.clip((next_lower - blur_map) / weight_range + 1, 0, 1)

            result += blurred_arr * weight

        # Normalize
        result = np.clip(result, 0, 1)

        return result

    def apply_depth_warp(
        self,
        shadow: np.ndarray,
        depth_map: np.ndarray,
        strength: float = 0.5
    ) -> np.ndarray:
        """
        Warp shadow based on depth map for uneven surface rendering.
        Depth map: 0 = far, 255 = near
        """
        h, w = shadow.shape
        depth_h, depth_w = depth_map.shape

        # Resize depth map if needed
        if (depth_h, depth_w) != (h, w):
            depth_img = Image.fromarray(depth_map)
            depth_img = depth_img.resize((w, h), Image.BILINEAR)
            depth_map = np.array(depth_img, dtype=np.float32)
        else:
            depth_map = depth_map.astype(np.float32)

        # Normalize depth to 0-1
        depth_norm = depth_map / 255.0

        # Compute depth gradients for normal estimation
        grad_x = np.gradient(depth_norm, axis=1) * strength * 50
        grad_y = np.gradient(depth_norm, axis=0) * strength * 50

        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Warp coordinates based on depth gradients
        # Shadow bends away from elevated areas
        new_x = x_coords + grad_x
        new_y = y_coords + grad_y

        # Clip coordinates
        new_x = np.clip(new_x, 0, w - 1)
        new_y = np.clip(new_y, 0, h - 1)

        # Remap shadow using bilinear interpolation
        from scipy.ndimage import map_coordinates
        warped = map_coordinates(shadow, [new_y, new_x], order=1, mode='constant', cval=0)

        # Modulate shadow intensity by depth (shadows lighter on elevated areas)
        elevation_factor = 1 - depth_norm * 0.3
        warped = warped * elevation_factor

        return warped.astype(np.float32)

    def generate_shadow(
        self,
        foreground: Image.Image,
        background: Image.Image,
        depth_map: Optional[Image.Image] = None,
        foreground_position: Tuple[int, int] = (0, 0)
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """
        Generate complete shadow composite.

        Args:
            foreground: RGBA image with transparency
            background: RGB background image
            depth_map: Optional grayscale depth map for warping
            foreground_position: (x, y) position to place foreground on background

        Returns:
            (composite, shadow_only, mask_debug)
        """
        # Ensure correct modes
        if foreground.mode != 'RGBA':
            foreground = foreground.convert('RGBA')
        if background.mode != 'RGB':
            background = background.convert('RGB')

        bg_w, bg_h = background.size
        fg_w, fg_h = foreground.size
        pos_x, pos_y = foreground_position

        # Extract foreground mask
        fg_mask = self.extract_alpha_mask(foreground)

        # Create full-size mask on canvas
        full_mask = np.zeros((bg_h, bg_w), dtype=np.float32)

        # Place mask at position (handle bounds)
        y1 = max(0, pos_y)
        y2 = min(bg_h, pos_y + fg_h)
        x1 = max(0, pos_x)
        x2 = min(bg_w, pos_x + fg_w)

        src_y1 = max(0, -pos_y)
        src_y2 = src_y1 + (y2 - y1)
        src_x1 = max(0, -pos_x)
        src_x2 = src_x1 + (x2 - x1)

        full_mask[y1:y2, x1:x2] = fg_mask[src_y1:src_y2, src_x1:src_x2]

        # Generate contact shadow
        contact_shadow = self.create_contact_shadow(full_mask)

        # Generate cast shadow
        cast_shadow, blur_map = self.create_cast_shadow(full_mask, (bg_h, bg_w))

        # Apply variable blur to cast shadow
        cast_shadow_blurred = self.apply_variable_blur(cast_shadow, blur_map)

        # Combine shadows
        combined_shadow = np.maximum(contact_shadow, cast_shadow_blurred)

        # Apply depth warping if depth map provided
        if depth_map is not None:
            depth_arr = np.array(depth_map.convert('L'))
            combined_shadow = self.apply_depth_warp(combined_shadow, depth_arr)

        # Apply shadow intensity
        combined_shadow = combined_shadow * self.shadow_intensity

        # Remove shadow under the foreground object
        combined_shadow = combined_shadow * (1 - full_mask * 0.95)

        # Create shadow-only image (for debug)
        shadow_only = Image.fromarray((combined_shadow * 255).astype(np.uint8), mode='L')

        # Create mask debug image
        mask_debug = Image.fromarray((full_mask * 255).astype(np.uint8), mode='L')

        # Create composite
        bg_array = np.array(background, dtype=np.float32)

        # Apply shadow to background (darken)
        shadow_3ch = np.stack([combined_shadow] * 3, axis=-1)
        shadowed_bg = bg_array * (1 - shadow_3ch * 0.8)  # Darken by shadow

        composite = Image.fromarray(shadowed_bg.astype(np.uint8), mode='RGB')

        # Paste foreground
        composite.paste(foreground, foreground_position, foreground)

        return composite, shadow_only, mask_debug


def auto_cutout_subject(image: Image.Image) -> Image.Image:
    """
    Simple background removal using edge detection and flood fill.
    For production, use rembg or similar ML-based solution.
    """
    try:
        from rembg import remove
        return remove(image)
    except ImportError:
        # Fallback: simple threshold-based removal (assumes light background)
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        arr = np.array(image)
        rgb = arr[:, :, :3]

        # Simple: assume corners are background color
        corners = [rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]]
        bg_color = np.mean(corners, axis=0)

        # Distance from background color
        diff = np.sqrt(np.sum((rgb.astype(float) - bg_color) ** 2, axis=-1))

        # Create alpha based on difference
        alpha = np.clip(diff / 100, 0, 1) * 255

        # Apply some morphological operations
        from scipy import ndimage
        alpha = ndimage.binary_erosion(alpha > 128, iterations=2)
        alpha = ndimage.binary_dilation(alpha, iterations=3)
        alpha = ndimage.gaussian_filter(alpha.astype(float), sigma=1) * 255

        arr[:, :, 3] = alpha.astype(np.uint8)
        return Image.fromarray(arr)


# Convenience function
def generate_shadow_composite(
    foreground_path: str,
    background_path: str,
    output_dir: str = "./output",
    depth_map_path: Optional[str] = None,
    light_angle: float = 135.0,
    light_elevation: float = 45.0,
    shadow_intensity: float = 0.7,
    position: Optional[Tuple[int, int]] = None,
    auto_cutout: bool = False
) -> Tuple[str, str, str]:
    """
    High-level function to generate shadow composite from file paths.
    Returns paths to (composite, shadow_only, mask_debug).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load images
    foreground = Image.open(foreground_path)
    background = Image.open(background_path)

    # Auto cutout if requested
    if auto_cutout:
        foreground = auto_cutout_subject(foreground)

    # Load depth map if provided
    depth_map = None
    if depth_map_path:
        depth_map = Image.open(depth_map_path)

    # Auto-position: center horizontally, bottom vertically
    if position is None:
        bg_w, bg_h = background.size
        fg_w, fg_h = foreground.size
        position = ((bg_w - fg_w) // 2, bg_h - fg_h - 20)

    # Generate shadow
    generator = ShadowGenerator(
        light_angle=light_angle,
        light_elevation=light_elevation,
        shadow_intensity=shadow_intensity
    )

    composite, shadow_only, mask_debug = generator.generate_shadow(
        foreground, background, depth_map, position
    )

    # Save outputs
    composite_path = os.path.join(output_dir, "composite.png")
    shadow_path = os.path.join(output_dir, "shadow_only.png")
    mask_path = os.path.join(output_dir, "mask_debug.png")

    composite.save(composite_path)
    shadow_only.save(shadow_path)
    mask_debug.save(mask_path)

    return composite_path, shadow_path, mask_path
