from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from scipy import ndimage
import os
import sys
import math

# Add python folder to path so we can import shadow_generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
from shadow_generator import ShadowGenerator

print('=== Realistic Shadow Generator ===')
print()

# Load images
foreground_orig = Image.open('samples/user_foreground.JPG').convert('RGBA')
background = Image.open('samples/user_background.JPG').convert('RGB')

print(f'Foreground: {foreground_orig.size}')
print(f'Background: {background.size}')

# ============================================
# STEP 1: Better background removal
# ============================================
print()
print('Step 1: Removing gray backdrop...')

arr = np.array(foreground_orig)
rgb = arr[:, :, :3].astype(np.float32)
h, w = rgb.shape[:2]

# Convert to HSV-like values for better segmentation
r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
max_rgb = np.maximum(np.maximum(r, g), b)
min_rgb = np.minimum(np.minimum(r, g), b)

# Saturation (0 for gray, high for colorful)
saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1), 0)

# Value/brightness
value = max_rgb / 255.0

# Sample the corners and edges to find background color
edge_size = 50
bg_samples = []
# Corners
bg_samples.extend(rgb[:edge_size, :edge_size].reshape(-1, 3))
bg_samples.extend(rgb[:edge_size, -edge_size:].reshape(-1, 3))
bg_samples.extend(rgb[-edge_size:, :edge_size].reshape(-1, 3))
bg_samples.extend(rgb[-edge_size:, -edge_size:].reshape(-1, 3))

bg_color = np.median(bg_samples, axis=0)
print(f'Detected background color (RGB): {bg_color.astype(int)}')

# Distance from background color
color_dist = np.sqrt(np.sum((rgb - bg_color) ** 2, axis=-1))

# Adaptive threshold - be more conservative to avoid cutting into the subject
threshold = 35  # Pixels within this distance from bg_color are background

# Create foreground mask
fg_mask = color_dist > threshold

# Don't use saturation filter as it cuts into skin/clothing
# The color distance alone should work for this gray backdrop

# Clean up the mask
print('Cleaning up mask...')

# Fill holes in foreground
fg_mask = ndimage.binary_fill_holes(fg_mask)

# Remove small noise (isolated pixels)
fg_mask = ndimage.binary_opening(fg_mask, iterations=2)

# Close small gaps
fg_mask = ndimage.binary_closing(fg_mask, iterations=8)

# Find the largest connected component (the person)
labeled, num_features = ndimage.label(fg_mask)
if num_features > 0:
    sizes = ndimage.sum(fg_mask, labeled, range(1, num_features + 1))
    largest = np.argmax(sizes) + 1
    fg_mask = labeled == largest
    print(f'Kept largest connected component (removed {num_features - 1} artifacts)')

# Smooth edges
alpha = ndimage.gaussian_filter(fg_mask.astype(np.float32), sigma=1.5)
alpha = np.clip(alpha, 0, 1)

# Apply alpha to foreground
arr[:, :, 3] = (alpha * 255).astype(np.uint8)
foreground_cutout = Image.fromarray(arr)

# Crop to actual content bounds to remove any edge artifacts
alpha_for_crop = np.array(foreground_cutout.split()[-1])
rows = np.any(alpha_for_crop > 10, axis=1)
cols = np.any(alpha_for_crop > 10, axis=0)
if rows.any() and cols.any():
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    foreground_cutout = foreground_cutout.crop((x1, y1, x2+1, y2+1))
    print(f'Cropped foreground to content bounds: {foreground_cutout.size}')

# ============================================
# STEP 2: Scale and position
# ============================================
print()
print('Step 2: Scaling and positioning...')

bg_w, bg_h = background.size
fg_w, fg_h = foreground_cutout.size

# Scale to fit nicely (about 45% of background height)
target_height = int(bg_h * 0.45)
scale = target_height / fg_h
new_w = int(fg_w * scale)
new_h = int(fg_h * scale)

foreground_scaled = foreground_cutout.resize((new_w, new_h), Image.LANCZOS)
print(f'Scaled to: {new_w}x{new_h}')

# Position: centered horizontally, leave MORE room at bottom for visible shadow
pos_x = (bg_w - new_w) // 2  # Centered
pos_y = bg_h - new_h - int(new_h * 0.35)  # Leave 35% height below for clear shadow

print(f'Position: ({pos_x}, {pos_y})')

# ============================================
# STEP 3: Generate realistic shadow using ShadowGenerator
# ============================================
print()
print('Step 3: Generating realistic shadow with ShadowGenerator...')

# Create shadow generator - light from above/behind for shadow directly below
# Light angle: 180 = light from behind, casting shadow in front (below child)
generator = ShadowGenerator(
    light_angle=180.0,          # Light from behind - shadow in front
    light_elevation=45.0,       # Higher angle = shorter, more visible shadow
    shadow_intensity=1.0,       # Maximum shadow intensity
    contact_shadow_size=40,     # Very large contact shadow
    max_blur=2.0,               # Minimal blur for sharp edges
    shadow_length_factor=0.8    # Shorter but more visible
)

print(f'Light angle: 45° (upper-right)')
print(f'Light elevation: 35°')
print(f'Shadow intensity: 0.8')

# Generate shadow composite using the ShadowGenerator
composite, shadow_only_img, mask_debug_img = generator.generate_shadow(
    foreground=foreground_scaled,
    background=background,
    depth_map=None,
    foreground_position=(pos_x, pos_y)
)

print('Shadow generation complete!')

# ============================================
# Save outputs
# ============================================
print()
print('Step 4: Saving outputs...')

os.makedirs('output', exist_ok=True)

# Save composite (PNG doesn't use quality parameter)
composite.save('output/composite.png')

# Save foreground cutout
foreground_scaled.save('output/foreground_cutout.png')

# Save shadow visualization (already a PIL Image from generator)
shadow_only_img.save('output/shadow_only.png')

# Save mask debug (already a PIL Image from generator)
mask_debug_img.save('output/mask_debug.png')

print()
print('=== Done! ===')
print('Output files in ./output/:')
print('  - composite.png (final result with realistic shadow)')
print('  - foreground_cutout.png (extracted person)')
print('  - shadow_only.png (shadow layer visualization)')
print('  - mask_debug.png (person mask)')
