#!/usr/bin/env python3
"""
Realistic Shadow Generator - Command Line Interface
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shadow_generator import generate_shadow_composite, ShadowGenerator, auto_cutout_subject
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Generate realistic shadows for image compositing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python cli.py -f person.png -b street.jpg

  # With custom light settings
  python cli.py -f person.png -b street.jpg --angle 45 --elevation 30

  # With depth map for surface warping
  python cli.py -f person.png -b street.jpg -d depth.png

  # Auto cutout foreground subject
  python cli.py -f photo.jpg -b background.jpg --auto-cutout

  # Custom position
  python cli.py -f person.png -b street.jpg --pos-x 100 --pos-y 200
        """
    )

    # Required arguments
    parser.add_argument(
        "-f", "--foreground",
        required=True,
        help="Path to foreground image (PNG with transparency, or use --auto-cutout)"
    )
    parser.add_argument(
        "-b", "--background",
        required=True,
        help="Path to background image"
    )

    # Optional arguments
    parser.add_argument(
        "-d", "--depth-map",
        default=None,
        help="Path to depth map (grayscale 0-255) for surface warping"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="./output",
        help="Output directory (default: ./output)"
    )

    # Light settings
    parser.add_argument(
        "--angle",
        type=float,
        default=135.0,
        help="Light angle in degrees (0-360, default: 135)"
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=45.0,
        help="Light elevation in degrees (0-90, default: 45)"
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=0.7,
        help="Shadow intensity (0-1, default: 0.7)"
    )

    # Position
    parser.add_argument(
        "--pos-x",
        type=int,
        default=None,
        help="X position for foreground (default: centered)"
    )
    parser.add_argument(
        "--pos-y",
        type=int,
        default=None,
        help="Y position for foreground (default: bottom)"
    )

    # Processing options
    parser.add_argument(
        "--auto-cutout",
        action="store_true",
        help="Automatically remove background from foreground image"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.foreground):
        print(f"Error: Foreground image not found: {args.foreground}")
        sys.exit(1)

    if not os.path.exists(args.background):
        print(f"Error: Background image not found: {args.background}")
        sys.exit(1)

    if args.depth_map and not os.path.exists(args.depth_map):
        print(f"Error: Depth map not found: {args.depth_map}")
        sys.exit(1)

    # Validate ranges
    if not 0 <= args.angle <= 360:
        print("Error: Light angle must be between 0 and 360")
        sys.exit(1)

    if not 0 <= args.elevation <= 90:
        print("Error: Light elevation must be between 0 and 90")
        sys.exit(1)

    if not 0 <= args.intensity <= 1:
        print("Error: Shadow intensity must be between 0 and 1")
        sys.exit(1)

    # Determine position
    position = None
    if args.pos_x is not None or args.pos_y is not None:
        bg = Image.open(args.background)
        fg = Image.open(args.foreground)
        bg_w, bg_h = bg.size
        fg_w, fg_h = fg.size

        pos_x = args.pos_x if args.pos_x is not None else (bg_w - fg_w) // 2
        pos_y = args.pos_y if args.pos_y is not None else (bg_h - fg_h - 20)
        position = (pos_x, pos_y)

    # Generate shadow composite
    print(f"Generating shadow composite...")
    print(f"  Light angle: {args.angle} degrees")
    print(f"  Light elevation: {args.elevation} degrees")
    print(f"  Shadow intensity: {args.intensity}")

    if args.depth_map:
        print(f"  Using depth map: {args.depth_map}")

    if args.auto_cutout:
        print(f"  Auto-cutout enabled")

    try:
        composite_path, shadow_path, mask_path = generate_shadow_composite(
            foreground_path=args.foreground,
            background_path=args.background,
            output_dir=args.output_dir,
            depth_map_path=args.depth_map,
            light_angle=args.angle,
            light_elevation=args.elevation,
            shadow_intensity=args.intensity,
            position=position,
            auto_cutout=args.auto_cutout
        )

        print(f"\nOutput files generated:")
        print(f"  Composite: {composite_path}")
        print(f"  Shadow only: {shadow_path}")
        print(f"  Mask debug: {mask_path}")

    except Exception as e:
        print(f"Error generating shadow: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
