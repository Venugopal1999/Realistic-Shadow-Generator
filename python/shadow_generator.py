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

        The shadow extends along the ground plane in the direction opposite to the light.
        In image coordinates: +x = right, +y = down.

        Light angle convention:
        - 0° = light from right
        - 90° = light from top
        - 180° = light from left
        - 270° = light from bottom
        """
        # Shadow length inversely related to elevation (lower sun = longer shadow)
        # At 90° elevation (directly overhead), shadow length approaches 0
        # At 0° elevation (horizon), shadow is very long
        if self.light_elevation >= 89:
            shadow_length = height * 0.1  # Minimal shadow when light is overhead
        else:
            shadow_length = height * self.shadow_length_factor * math.tan(math.radians(90 - self.light_elevation))
        shadow_length = min(shadow_length, height * 2)  # Cap at 2x height for reasonable shadow

        # Shadow direction is opposite to light direction
        # The shadow falls on the ground (horizontal plane), so we compute
        # the horizontal component of the shadow direction
        shadow_angle = self.light_angle + 180  # Opposite direction
        angle_rad = math.radians(shadow_angle)

        # dx: horizontal offset (positive = right, negative = left)
        dx = shadow_length * math.cos(angle_rad)

        # dy: In image coords, positive y is DOWN. For a shadow on the ground,
        # dy should be small and positive (shadow appears below/in front of object)
        # We use a small vertical spread based on viewing angle (assume ~30° from horizontal)
        dy = shadow_length * 0.2  # Small downward component to place shadow on ground

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
        Create a clearly visible cast shadow using PIL transform.
        Returns (shadow_mask, blur_map) for variable blur application.
        """
        canvas_h, canvas_w = canvas_size

        # Create output arrays
        shadow = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        blur_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        # Find object bounds
        rows = np.any(mask > 0.1, axis=1)
        cols = np.any(mask > 0.1, axis=0)

        if not rows.any() or not cols.any():
            return shadow, blur_map

        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        object_height = row_max - row_min
        object_base = row_max
        object_center_x = (col_min + col_max) // 2

        # Shadow parameters
        shadow_angle = self.light_angle + 180
        angle_rad = math.radians(shadow_angle)

        if self.light_elevation >= 89:
            shadow_length = object_height * 0.1
        else:
            shadow_length = object_height * math.tan(math.radians(90 - self.light_elevation)) * self.shadow_length_factor
        shadow_length = min(shadow_length, object_height * 2.5)

        # Extract object mask as image for transformation
        obj_mask = mask[row_min:row_max+1, col_min:col_max+1]
        obj_h, obj_w = obj_mask.shape

        # FLIP the mask vertically so head projects to far end of shadow
        # (top of person = head should be at far end, bottom = feet at near end)
        obj_mask_flipped = np.flipud(obj_mask)

        # Convert to PIL for transform
        mask_img = Image.fromarray((obj_mask_flipped * 255).astype(np.uint8), mode='L')

        # Calculate shear transform coefficients
        # PIL transform uses: x' = ax + by + c, y' = dx + ey + f
        # For shadow: we want to shear horizontally and compress vertically

        shear_amount = math.cos(angle_rad) * shadow_length / obj_h
        vertical_scale = 0.35  # Compress to 35% height for ground projection

        # New dimensions
        new_h = int(obj_h * vertical_scale) + 20
        new_w = obj_w + int(abs(shear_amount) * obj_h) + 20

        # Create transform using AFFINE
        # The coefficients map output to input: input = a*x + b*y + c, etc.
        # We want: src_x = dst_x - shear * dst_y, src_y = dst_y / vertical_scale

        if shear_amount >= 0:
            x_shift = 0
        else:
            x_shift = int(abs(shear_amount) * obj_h)

        # Affine coefficients (a, b, c, d, e, f) where:
        # src_x = a*dst_x + b*dst_y + c
        # src_y = d*dst_x + e*dst_y + f
        a = 1
        b = -shear_amount * (1 / vertical_scale)
        c = -x_shift
        d = 0
        e = 1 / vertical_scale
        f = 0

        try:
            shadow_transformed = mask_img.transform(
                (new_w, new_h),
                Image.AFFINE,
                (a, b, c, d, e, f),
                resample=Image.BILINEAR
            )
        except Exception:
            # Fallback: just use original mask
            shadow_transformed = mask_img.resize((new_w, new_h), Image.BILINEAR)

        # Convert back to numpy
        shadow_arr = np.array(shadow_transformed, dtype=np.float32) / 255.0

        # Create intensity gradient (stronger at base)
        gradient = np.linspace(1.0, 0.6, new_h).reshape(-1, 1)
        shadow_arr = shadow_arr * gradient

        # Fill holes and clean up
        if np.any(shadow_arr > 0.05):
            shadow_binary = shadow_arr > 0.05
            shadow_binary = ndimage.binary_fill_holes(shadow_binary)
            shadow_binary = ndimage.binary_closing(shadow_binary, iterations=8)
            shadow_binary = ndimage.binary_dilation(shadow_binary, iterations=3)

            # Re-apply gradient to filled shadow - use high intensity for clear edges
            shadow_arr = np.where(shadow_binary, 1.0, 0) * gradient

            # Very light smoothing for clean but sharp edges
            shadow_arr = ndimage.gaussian_filter(shadow_arr, sigma=1)

        # Create blur map
        blur_arr = np.linspace(0, 1, new_h).reshape(-1, 1) * np.ones((1, new_w))
        blur_arr = blur_arr * (shadow_arr > 0.1).astype(np.float32)

        # Place on canvas
        place_y = object_base
        place_x = col_min - x_shift

        # Copy to canvas
        for sy in range(new_h):
            cy = place_y + sy
            if 0 <= cy < canvas_h:
                for sx in range(new_w):
                    cx = place_x + sx
                    if 0 <= cx < canvas_w:
                        shadow[cy, cx] = max(shadow[cy, cx], shadow_arr[sy, sx])
                        blur_map[cy, cx] = max(blur_map[cy, cx], blur_arr[sy, sx])

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

        # Apply shadow to background (darken) - VERY dark shadow for clear visibility
        # combined_shadow is 0-1, we want to make shadow areas very dark
        shadow_factor = (1 - combined_shadow * 1.5).astype(np.float32)  # Boost shadow effect
        shadow_factor = np.clip(shadow_factor, 0.0, 1.0)  # Allow completely black
        for c in range(3):
            bg_array[:, :, c] *= shadow_factor

        composite = Image.fromarray(bg_array.astype(np.uint8), mode='RGB')

        # Paste foreground
        composite.paste(foreground, foreground_position, foreground)

        return composite, shadow_only, mask_debug


def auto_cutout_subject(image: Image.Image) -> Image.Image:
    """
    Background removal using color-based segmentation.
    For production, use rembg or similar ML-based solution.
    """
    try:
        from rembg import remove
        return remove(image)
    except ImportError:
        # Fallback: improved color-based background removal
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        arr = np.array(image)
        rgb = arr[:, :, :3].astype(np.float32)
        h, w = rgb.shape[:2]

        # Sample background color from edges (more robust than just corners)
        edge_samples = []
        sample_size = 20  # pixels from edge
        # Top edge
        edge_samples.extend(rgb[:sample_size, :].reshape(-1, 3).tolist())
        # Bottom edge
        edge_samples.extend(rgb[-sample_size:, :].reshape(-1, 3).tolist())
        # Left edge
        edge_samples.extend(rgb[:, :sample_size].reshape(-1, 3).tolist())
        # Right edge
        edge_samples.extend(rgb[:, -sample_size:].reshape(-1, 3).tolist())

        # Use median for robustness against outliers
        bg_color = np.median(edge_samples, axis=0)

        # Color distance from background (Euclidean in RGB space)
        diff = np.sqrt(np.sum((rgb - bg_color) ** 2, axis=-1))

        # Adaptive threshold based on image statistics
        threshold = max(30, np.percentile(diff, 30))

        # Create binary mask
        foreground_mask = diff > threshold

        # Morphological operations to clean up mask
        from scipy import ndimage

        # Fill holes in the foreground
        foreground_mask = ndimage.binary_fill_holes(foreground_mask)

        # Remove small noise
        foreground_mask = ndimage.binary_opening(foreground_mask, iterations=2)

        # Close small gaps
        foreground_mask = ndimage.binary_closing(foreground_mask, iterations=5)

        # Dilate slightly to include edges
        foreground_mask = ndimage.binary_dilation(foreground_mask, iterations=2)

        # Smooth edges
        alpha = ndimage.gaussian_filter(foreground_mask.astype(np.float32), sigma=2)
        alpha = np.clip(alpha * 1.5, 0, 1)  # Boost and clip

        arr[:, :, 3] = (alpha * 255).astype(np.uint8)
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

    # Auto-position: center horizontally, positioned to leave room for shadow
    if position is None:
        bg_w, bg_h = background.size
        fg_w, fg_h = foreground.size
        # Leave more vertical space below for shadow (at least 25% of foreground height)
        shadow_margin = max(int(fg_h * 0.25), 50)
        position = ((bg_w - fg_w) // 2, bg_h - fg_h - shadow_margin)

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
