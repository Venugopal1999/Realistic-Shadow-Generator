# Realistic Shadow Generator

A compositing tool that generates realistic shadows when placing a foreground subject onto a background image. Implements physically-inspired shadow rendering with directional light control, contact shadows, and depth-aware warping.

## Features

### Core Features
- **Directional Light Control**: Adjust light angle (0-360) and elevation (0-90)
- **Contact Shadows**: Dark, sharp shadows near the contact area that fade with distance
- **Soft Shadow Falloff**: Blur increases as shadow extends from contact point
- **Silhouette-Matching Shadows**: True projected shadows from subject shape, not simple ovals

### Bonus Feature
- **Depth Map Warping**: When provided with a depth map, shadows bend and warp to match uneven surfaces

## Implementations

This project includes **both** Python and TypeScript implementations:

| Feature | Python | TypeScript/Web |
|---------|--------|----------------|
| CLI Interface | Yes | - |
| GUI | PyQt6 | HTML/CSS |
| Core Algorithm | NumPy/SciPy | Canvas API |
| Depth Warping | Yes | Yes |

---

## Python Version

### Installation

```bash
cd python
pip install -r requirements.txt
```

Required packages:
- `numpy>=1.20.0`
- `Pillow>=9.0.0`
- `scipy>=1.7.0`
- `PyQt6>=6.4.0` (optional, for GUI)

### CLI Usage

```bash
# Basic usage
python cli.py -f foreground.png -b background.jpg

# With custom light settings
python cli.py -f foreground.png -b background.jpg --angle 45 --elevation 30

# With depth map for surface warping (bonus feature)
python cli.py -f foreground.png -b background.jpg -d depth.png

# Full options
python cli.py \
    -f foreground.png \
    -b background.jpg \
    -d depth.png \
    -o ./output \
    --angle 135 \
    --elevation 45 \
    --intensity 0.7 \
    --pos-x 100 \
    --pos-y 200
```

#### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-f, --foreground` | Foreground image (PNG with transparency) | Required |
| `-b, --background` | Background image | Required |
| `-d, --depth-map` | Grayscale depth map (0-255) | Optional |
| `-o, --output-dir` | Output directory | `./output` |
| `--angle` | Light angle in degrees (0-360) | 135 |
| `--elevation` | Light elevation in degrees (0-90) | 45 |
| `--intensity` | Shadow intensity (0-1) | 0.7 |
| `--pos-x` | X position for foreground | Centered |
| `--pos-y` | Y position for foreground | Bottom |
| `--auto-cutout` | Auto-remove foreground background | False |

### PyQt6 GUI

```bash
python ui.py
```

The GUI provides:
- Drag-and-drop image loading
- Real-time slider controls for light settings
- Live preview of composite, shadow, and mask
- One-click export of all outputs

---

## TypeScript/Web Version

### Installation

```bash
cd web
npm install
npm run build
```

### Running

```bash
npm run serve
```

Then open `http://localhost:3000` in your browser.

### Development

```bash
npm run watch  # Watch for TypeScript changes
```

### Web Features
- Drag-and-drop image upload
- Real-time parameter adjustment
- Instant preview
- Download all outputs (composite, shadow, mask)

---

## Output Files

The generator produces three files:

| File | Description |
|------|-------------|
| `composite.png` | Final composite with shadow applied |
| `shadow_only.png` | Shadow layer only (for debugging) |
| `mask_debug.png` | Foreground mask (for debugging) |

---

## How It Works

### Shadow Generation Algorithm

1. **Extract Alpha Mask**: Get the subject silhouette from the foreground image's alpha channel

2. **Compute Shadow Offset**: Calculate projection direction based on light angle and elevation
   - Lower elevation = longer shadows
   - Angle determines shadow direction (opposite to light source)

3. **Create Contact Shadow**: Generate ambient occlusion near the subject's base
   - Sharp and dark at contact point
   - Quick falloff with distance

4. **Create Cast Shadow**: Project subject silhouette with perspective
   - Points higher up cast shadows further away
   - Shadow intensity decreases with distance

5. **Apply Variable Blur**: Blur amount increases with distance from contact
   - Sharp near feet/base
   - Progressively softer at shadow tip

6. **Depth Warping** (if depth map provided):
   - Compute surface normals from depth gradients
   - Warp shadow coordinates based on surface orientation
   - Modulate shadow intensity by elevation

7. **Composite**: Blend shadow onto background, then overlay foreground

### Light Model

```
Shadow Direction = Light Angle + 180
Shadow Length = Height * tan(90 - Elevation) * Factor
```

- Light Angle: 0 = right, 90 = bottom, 180 = left, 270 = top
- Light Elevation: 0 = horizon (long shadows), 90 = overhead (no shadow)

---

## Sample Usage

Generate sample images:
```bash
python create_samples.py
```

This creates:
- `samples/foreground.png` - Person silhouette with transparency
- `samples/background.png` - Simple outdoor scene
- `samples/depth_map.png` - Grayscale depth map for ground plane

Run the generator:
```bash
python python/cli.py -f samples/foreground.png -b samples/background.png -o output
```

---

## Architecture

```
shadow-generator/
    python/
        shadow_generator.py   # Core algorithm
        cli.py                 # Command-line interface
        ui.py                  # PyQt6 GUI
        requirements.txt
    web/
        src/
            shadow-generator.ts  # Core algorithm (TypeScript)
            index.ts              # Web app entry point
            styles.css            # UI styles
        index.html
        package.json
        tsconfig.json
    samples/                   # Sample images
    output/                    # Generated outputs
    create_samples.py          # Sample image generator
    README.md
```

---

## Technical Notes

### Performance Considerations
- Variable blur uses multi-pass approach (8 blur levels blended)
- Shadow projection iterates over object bounding box only
- Web version uses pure Canvas API (no WebGL) for compatibility

### Limitations
- No soft penumbra simulation (single light source assumed)
- Depth warping is approximate (not ray-traced)
- Contact shadow is simplified (no true ambient occlusion)

### Future Improvements
- Multiple light sources
- Colored shadows
- GPU acceleration (WebGL/CUDA)
- ML-based background removal (rembg integration)

---

## License

MIT License
