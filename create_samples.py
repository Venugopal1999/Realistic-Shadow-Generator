#!/usr/bin/env python3
"""
Create sample test images for the Shadow Generator demo.
This generates a simple foreground silhouette, background, and depth map.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os

def create_person_silhouette(width=200, height=400):
    """Create a simple person-like silhouette with transparency."""
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Body color
    color = (50, 80, 120, 255)

    cx = width // 2

    # Head
    head_radius = 30
    head_y = 50
    draw.ellipse([cx - head_radius, head_y - head_radius,
                  cx + head_radius, head_y + head_radius], fill=color)

    # Neck
    draw.rectangle([cx - 12, head_y + head_radius - 5, cx + 12, head_y + head_radius + 20], fill=color)

    # Torso
    torso_top = head_y + head_radius + 15
    torso_bottom = torso_top + 120
    draw.polygon([
        (cx - 50, torso_top),
        (cx + 50, torso_top),
        (cx + 40, torso_bottom),
        (cx - 40, torso_bottom)
    ], fill=color)

    # Arms
    arm_width = 18
    # Left arm
    draw.polygon([
        (cx - 50, torso_top + 5),
        (cx - 50 - arm_width, torso_top + 5),
        (cx - 65, torso_top + 100),
        (cx - 45, torso_top + 100)
    ], fill=color)

    # Right arm
    draw.polygon([
        (cx + 50, torso_top + 5),
        (cx + 50 + arm_width, torso_top + 5),
        (cx + 65, torso_top + 100),
        (cx + 45, torso_top + 100)
    ], fill=color)

    # Legs
    leg_top = torso_bottom - 5
    leg_bottom = height - 10

    # Left leg
    draw.polygon([
        (cx - 35, leg_top),
        (cx - 5, leg_top),
        (cx - 8, leg_bottom),
        (cx - 35, leg_bottom)
    ], fill=color)

    # Right leg
    draw.polygon([
        (cx + 5, leg_top),
        (cx + 35, leg_top),
        (cx + 35, leg_bottom),
        (cx + 8, leg_bottom)
    ], fill=color)

    # Smooth the edges
    img = img.filter(ImageFilter.GaussianBlur(1))

    # Restore alpha edges
    arr = np.array(img)
    arr[:, :, 3] = np.where(arr[:, :, 3] > 30, 255, 0)
    img = Image.fromarray(arr)

    return img


def create_background(width=800, height=600):
    """Create a simple gradient background with a ground plane."""
    img = Image.new('RGB', (width, height))
    arr = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient (blue to lighter blue)
    horizon = int(height * 0.6)
    for y in range(horizon):
        t = y / horizon
        r = int(135 + (200 - 135) * t)
        g = int(180 + (220 - 180) * t)
        b = int(250 + (255 - 250) * t)
        arr[y, :] = [r, g, b]

    # Ground (green-ish gray)
    for y in range(horizon, height):
        t = (y - horizon) / (height - horizon)
        r = int(120 - 20 * t)
        g = int(140 - 30 * t)
        b = int(100 - 20 * t)
        arr[y, :] = [r, g, b]

    img = Image.fromarray(arr)

    # Add some texture noise
    noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
    arr = np.clip(np.array(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    return img


def create_depth_map(width=800, height=600):
    """Create a depth map where ground recedes into distance."""
    arr = np.zeros((height, width), dtype=np.uint8)

    horizon = int(height * 0.6)

    # Sky is far (low values)
    arr[:horizon, :] = 30

    # Ground gets closer as it goes down
    for y in range(horizon, height):
        t = (y - horizon) / (height - horizon)
        depth = int(30 + (255 - 30) * t)  # Far to near
        arr[y, :] = depth

    img = Image.fromarray(arr, mode='L')
    return img


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'samples')
    os.makedirs(output_dir, exist_ok=True)

    print("Creating sample images...")

    # Create foreground
    foreground = create_person_silhouette(200, 400)
    fg_path = os.path.join(output_dir, 'foreground.png')
    foreground.save(fg_path)
    print(f"  Created: {fg_path}")

    # Create background
    background = create_background(800, 600)
    bg_path = os.path.join(output_dir, 'background.png')
    background.save(bg_path)
    print(f"  Created: {bg_path}")

    # Create depth map
    depth = create_depth_map(800, 600)
    depth_path = os.path.join(output_dir, 'depth_map.png')
    depth.save(depth_path)
    print(f"  Created: {depth_path}")

    print("\nSample images created successfully!")
    print(f"Location: {output_dir}")


if __name__ == "__main__":
    main()
