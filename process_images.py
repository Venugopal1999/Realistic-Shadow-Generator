#!/usr/bin/env python3
"""
Process user images with background removal and shadow generation.
"""

import numpy as np
from PIL import Image, ImageFilter
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
from shadow_generator import ShadowGenerator


def remove_gray_background(image: Image.Image) -> Image.Image:
    """
    Remove gray background from a studio photo.
    Specifically tuned for photos with uniform gray/dark backgrounds.
    """
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    arr = np.array(image, dtype=np.float32)
    rgb = arr[:, :, :3]
    h, w = rgb.shape[:2]

    # Sample background color from edges (top and sides)
    edge_pixels = []

    # Top edge
    for x in range(0, w, 5):
        for y in range(0, min(50, h)):
            edge_pixels.append(rgb[y, x])

    # Left edge
    for y in range(0, h, 5):
        for x in range(0, min(50, w)):
            edge_pixels.append(rgb[y, x])

    # Right edge
    for y in range(0, h, 5):
        for x in range(max(0, w-50), w):
            edge_pixels.append(rgb[y, x])

    edge_pixels = np.array(edge_pixels)
    bg_color = np.median(edge_pixels, axis=0)

    print(f"Detected background color: RGB({bg_color[0]:.0f}, {bg_color[1]:.0f}, {bg_color[2]:.0f})")

    # Calculate distance from background color for each pixel
    diff = rgb - bg_color
    dist_from_bg = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Create alpha mask - pixels FAR from background get high alpha (keep them)
    # Pixels CLOSE to background get low alpha (remove them)
    threshold = 50  # Distance threshold

    # Normalize distance to 0-1 range
    alpha = np.clip(dist_from_bg / threshold, 0, 1)

    # Make it more binary with a soft threshold
    alpha = np.where(dist_from_bg > threshold * 0.7, 1.0, alpha)
    alpha = np.where(dist_from_bg < threshold * 0.3, 0.0, alpha)

    # Clean up with morphological operations
    from scipy import ndimage

    # Convert to binary for morphological ops
    binary = alpha > 0.5

    # Fill holes in the subject
    binary = ndimage.binary_fill_holes(binary)

    # Remove small noise
    binary = ndimage.binary_opening(binary, iterations=3)

    # Smooth the edges
    binary = ndimage.binary_closing(binary, iterations=2)

    # Create smooth alpha from binary mask
    alpha_smooth = ndimage.gaussian_filter(binary.astype(float), sigma=2)

    # Ensure solid core
    alpha_final = np.where(binary, np.maximum(alpha_smooth, 0.95), alpha_smooth * 0.5)
    alpha_final = np.clip(alpha_final, 0, 1)

    # Create output with alpha channel
    result = arr.copy()
    result[:, :, 3] = (alpha_final * 255).astype(np.uint8)

    return Image.fromarray(result.astype(np.uint8))


def process_composite(fg_path: str, bg_path: str, output_dir: str = "./output"):
    """Process foreground and background to create shadow composite."""

    os.makedirs(output_dir, exist_ok=True)

    print("Loading images...")
    foreground = Image.open(fg_path)
    background = Image.open(bg_path)

    print(f"Foreground size: {foreground.size}")
    print(f"Background size: {background.size}")

    # Remove background from foreground
    print("Removing background from foreground...")
    fg_cutout = remove_gray_background(foreground)

    # Save the cutout for inspection
    cutout_path = os.path.join(output_dir, "foreground_cutout.png")
    fg_cutout.save(cutout_path)
    print(f"Saved cutout: {cutout_path}")

    # Resize foreground to fit nicely in background
    bg_w, bg_h = background.size
    fg_w, fg_h = fg_cutout.size

    # Scale foreground to be about 50% of background height
    target_height = int(bg_h * 0.55)
    scale = target_height / fg_h
    new_fg_w = int(fg_w * scale)
    new_fg_h = int(fg_h * scale)

    fg_resized = fg_cutout.resize((new_fg_w, new_fg_h), Image.LANCZOS)

    # Position: left side, standing on ground (the cobblestone area)
    pos_x = int(bg_w * 0.05)  # Left side
    pos_y = bg_h - new_fg_h - int(bg_h * 0.02)  # Near bottom

    print(f"Placing subject at position ({pos_x}, {pos_y})...")
    print(f"Subject size: {new_fg_w}x{new_fg_h}")

    # Generate shadow - light coming from upper right (matching the scene)
    generator = ShadowGenerator(
        light_angle=45,         # Light from upper-right
        light_elevation=35,     # Lower sun = longer shadow
        shadow_intensity=0.7,   # Stronger shadow
        contact_shadow_size=35,
        max_blur=50,
        shadow_length_factor=1.5
    )

    composite, shadow_only, mask_debug = generator.generate_shadow(
        fg_resized,
        background,
        None,  # No depth map
        (pos_x, pos_y)
    )

    # Save outputs
    composite_path = os.path.join(output_dir, "composite.png")
    shadow_path = os.path.join(output_dir, "shadow_only.png")
    mask_path = os.path.join(output_dir, "mask_debug.png")

    composite.save(composite_path)
    shadow_only.save(shadow_path)
    mask_debug.save(mask_path)

    print(f"\nOutput saved:")
    print(f"  Composite: {composite_path}")
    print(f"  Shadow: {shadow_path}")
    print(f"  Mask: {mask_path}")

    return composite_path, shadow_path, mask_path


if __name__ == "__main__":
    # Process the user's images (child is background file, car/city is foreground file)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fg_path = os.path.join(script_dir, "user_background.JPG")  # Child photo
    bg_path = os.path.join(script_dir, "user_foreground.JPG")  # Car/city scene

    if os.path.exists(fg_path) and os.path.exists(bg_path):
        output_dir = os.path.join(script_dir, "output")
        process_composite(fg_path, bg_path, output_dir)
    else:
        print(f"Images not found at {fg_path} or {bg_path}")
