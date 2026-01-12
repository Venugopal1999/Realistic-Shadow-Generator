/**
 * Realistic Shadow Generator - TypeScript/Canvas Implementation
 *
 * Generates physically-inspired shadows with:
 * - Directional light control (angle + elevation)
 * - Contact shadows (sharp near contact, fading with distance)
 * - Soft shadow falloff (blur increases with distance)
 * - Depth map warping (bonus)
 */

export interface ShadowConfig {
  lightAngle: number;      // 0-360 degrees
  lightElevation: number;  // 0-90 degrees
  shadowIntensity: number; // 0-1
  contactShadowSize: number;
  maxBlur: number;
  shadowLengthFactor: number;
}

export interface GenerationResult {
  composite: ImageData;
  shadowOnly: ImageData;
  maskDebug: ImageData;
}

export class ShadowGenerator {
  private config: ShadowConfig;

  constructor(config: Partial<ShadowConfig> = {}) {
    this.config = {
      lightAngle: config.lightAngle ?? 135,
      lightElevation: config.lightElevation ?? 45,
      shadowIntensity: config.shadowIntensity ?? 0.7,
      contactShadowSize: config.contactShadowSize ?? 15,
      maxBlur: config.maxBlur ?? 30,
      shadowLengthFactor: config.shadowLengthFactor ?? 1.5
    };
  }

  updateConfig(config: Partial<ShadowConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Extract alpha channel as a normalized float array
   */
  private extractAlphaMask(imageData: ImageData): Float32Array {
    const { width, height, data } = imageData;
    const mask = new Float32Array(width * height);

    for (let i = 0; i < width * height; i++) {
      mask[i] = data[i * 4 + 3] / 255;
    }

    return mask;
  }

  /**
   * Compute shadow offset based on light angle and elevation
   */
  private computeShadowOffset(height: number): { dx: number; dy: number } {
    const angleRad = (this.config.lightAngle * Math.PI) / 180;

    // Shadow length inversely related to elevation
    // At 90° elevation (overhead), shadow is minimal
    // At 0° elevation (horizon), shadow is very long
    const elevationFactor = Math.max(0.1, Math.cos(this.config.lightElevation * Math.PI / 180));
    let shadowLength = height * this.config.shadowLengthFactor * elevationFactor;
    shadowLength = Math.min(shadowLength, height * 2);

    // Shadow direction opposite to light
    const dx = shadowLength * Math.cos(angleRad + Math.PI);
    const dy = shadowLength * Math.sin(angleRad + Math.PI);

    console.log(`Shadow offset: dx=${dx.toFixed(1)}, dy=${dy.toFixed(1)}, length=${shadowLength.toFixed(1)}`);
    console.log(`Light settings: angle=${this.config.lightAngle}, elevation=${this.config.lightElevation}`);

    return { dx, dy };
  }

  /**
   * Apply Gaussian blur to a single-channel array
   */
  private gaussianBlur(
    data: Float32Array,
    width: number,
    height: number,
    radius: number
  ): Float32Array {
    if (radius < 1) return data.slice();

    const sigma = radius / 2;
    const kernelSize = Math.ceil(radius * 2) * 2 + 1;
    const kernel: number[] = [];
    let sum = 0;

    // Create 1D Gaussian kernel
    for (let i = 0; i < kernelSize; i++) {
      const x = i - Math.floor(kernelSize / 2);
      const g = Math.exp(-(x * x) / (2 * sigma * sigma));
      kernel.push(g);
      sum += g;
    }

    // Normalize kernel
    for (let i = 0; i < kernelSize; i++) {
      kernel[i] /= sum;
    }

    const halfKernel = Math.floor(kernelSize / 2);
    const temp = new Float32Array(width * height);
    const result = new Float32Array(width * height);

    // Horizontal pass
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let val = 0;
        for (let k = 0; k < kernelSize; k++) {
          const sx = Math.min(Math.max(x + k - halfKernel, 0), width - 1);
          val += data[y * width + sx] * kernel[k];
        }
        temp[y * width + x] = val;
      }
    }

    // Vertical pass
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let val = 0;
        for (let k = 0; k < kernelSize; k++) {
          const sy = Math.min(Math.max(y + k - halfKernel, 0), height - 1);
          val += temp[sy * width + x] * kernel[k];
        }
        result[y * width + x] = val;
      }
    }

    return result;
  }

  /**
   * Create contact shadow (ambient occlusion at base)
   */
  private createContactShadow(
    mask: Float32Array,
    width: number,
    height: number
  ): Float32Array {
    // Dilate the mask
    const dilated = new Float32Array(width * height);
    const dilateRadius = Math.floor(this.config.contactShadowSize / 2);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let maxVal = 0;
        for (let dy = -dilateRadius; dy <= dilateRadius; dy++) {
          for (let dx = -dilateRadius; dx <= dilateRadius; dx++) {
            const ny = y + dy;
            const nx = x + dx;
            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
              maxVal = Math.max(maxVal, mask[ny * width + nx]);
            }
          }
        }
        dilated[y * width + x] = maxVal;
      }
    }

    // Apply blur
    const blurred = this.gaussianBlur(dilated, width, height, 3);

    // Mask out foreground
    const contact = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
      contact[i] = blurred[i] * (1 - mask[i]) * 0.8;
    }

    return contact;
  }

  /**
   * Create cast shadow with perspective projection
   */
  private createCastShadow(
    mask: Float32Array,
    width: number,
    height: number,
    canvasWidth: number,
    canvasHeight: number,
    offsetX: number,
    offsetY: number
  ): { shadow: Float32Array; blurMap: Float32Array } {
    const shadow = new Float32Array(canvasWidth * canvasHeight);
    const blurMap = new Float32Array(canvasWidth * canvasHeight);

    // Find object bounds
    let rowMin = height, rowMax = 0;
    let colMin = width, colMax = 0;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (mask[y * width + x] > 0.1) {
          rowMin = Math.min(rowMin, y);
          rowMax = Math.max(rowMax, y);
          colMin = Math.min(colMin, x);
          colMax = Math.max(colMax, x);
        }
      }
    }

    if (rowMin > rowMax) return { shadow, blurMap };

    const objectHeight = rowMax - rowMin;
    const objectBase = rowMax;

    const { dx, dy } = this.computeShadowOffset(objectHeight);

    console.log(`Cast shadow: object height=${objectHeight}, base=${objectBase}, dx=${dx.toFixed(1)}, dy=${dy.toFixed(1)}`);

    // Project shadow - cast it on the ground below the object
    for (let y = rowMin; y <= rowMax; y++) {
      const distFromBase = objectBase - y;
      const progress = distFromBase / Math.max(objectHeight, 1);

      // Shadow offset increases with height from base
      const rowDx = dx * progress;
      const rowDy = dy * progress;

      for (let x = colMin; x <= colMax; x++) {
        const maskVal = mask[y * width + x];
        if (maskVal > 0.1) {
          // Project shadow point - starts at base and extends outward
          const shadowX = Math.round(x + offsetX + rowDx);
          const shadowY = Math.round(objectBase + offsetY + Math.abs(rowDy) + distFromBase * 0.2);

          if (shadowX >= 0 && shadowX < canvasWidth &&
              shadowY >= 0 && shadowY < canvasHeight) {
            const idx = shadowY * canvasWidth + shadowX;
            const intensity = maskVal * (1 - 0.4 * progress);
            shadow[idx] = Math.max(shadow[idx], intensity);
            blurMap[idx] = progress;
          }
        }
      }
    }

    console.log(`Shadow generated with ${shadow.filter(v => v > 0).length} non-zero pixels`);

    return { shadow, blurMap };
  }

  /**
   * Apply variable blur based on blur map
   */
  private applyVariableBlur(
    shadow: Float32Array,
    blurMap: Float32Array,
    width: number,
    height: number
  ): Float32Array {
    const blurLevels = 8;
    const result = new Float32Array(width * height);

    // Create blur levels
    const blurred: Float32Array[] = [];
    for (let i = 0; i < blurLevels; i++) {
      const progress = i / (blurLevels - 1);
      const radius = 1 + progress * this.config.maxBlur;
      blurred.push(this.gaussianBlur(shadow, width, height, radius));
    }

    // Blend based on blur map
    for (let i = 0; i < width * height; i++) {
      const targetLevel = blurMap[i] * (blurLevels - 1);
      const lowerLevel = Math.floor(targetLevel);
      const upperLevel = Math.min(lowerLevel + 1, blurLevels - 1);
      const t = targetLevel - lowerLevel;

      result[i] = blurred[lowerLevel][i] * (1 - t) + blurred[upperLevel][i] * t;
    }

    return result;
  }

  /**
   * Apply depth-based warping to shadow
   */
  private applyDepthWarp(
    shadow: Float32Array,
    depthMap: Float32Array,
    width: number,
    height: number,
    strength: number = 0.5
  ): Float32Array {
    const result = new Float32Array(width * height);

    // Compute gradients
    const gradX = new Float32Array(width * height);
    const gradY = new Float32Array(width * height);

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        gradX[idx] = (depthMap[idx + 1] - depthMap[idx - 1]) * strength * 50;
        gradY[idx] = (depthMap[idx + width] - depthMap[idx - width]) * strength * 50;
      }
    }

    // Warp shadow
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        const newX = Math.min(Math.max(Math.round(x + gradX[idx]), 0), width - 1);
        const newY = Math.min(Math.max(Math.round(y + gradY[idx]), 0), height - 1);

        const srcIdx = newY * width + newX;
        const elevationFactor = 1 - depthMap[idx] * 0.3;
        result[idx] = shadow[srcIdx] * elevationFactor;
      }
    }

    return result;
  }

  /**
   * Generate complete shadow composite
   */
  generate(
    foreground: ImageData,
    background: ImageData,
    depthMap: ImageData | null = null,
    position: { x: number; y: number } = { x: 0, y: 0 }
  ): GenerationResult {
    const { width: fgW, height: fgH } = foreground;
    const { width: bgW, height: bgH } = background;

    // Extract foreground mask
    const fgMask = this.extractAlphaMask(foreground);

    // Create full-size mask
    const fullMask = new Float32Array(bgW * bgH);
    for (let y = 0; y < fgH; y++) {
      const destY = y + position.y;
      if (destY < 0 || destY >= bgH) continue;

      for (let x = 0; x < fgW; x++) {
        const destX = x + position.x;
        if (destX < 0 || destX >= bgW) continue;

        fullMask[destY * bgW + destX] = fgMask[y * fgW + x];
      }
    }

    // Generate contact shadow
    const contactShadow = this.createContactShadow(fullMask, bgW, bgH);

    // Generate cast shadow
    const { shadow: castShadow, blurMap } = this.createCastShadow(
      fgMask, fgW, fgH, bgW, bgH, position.x, position.y
    );

    // Apply variable blur
    let castShadowBlurred = this.applyVariableBlur(castShadow, blurMap, bgW, bgH);

    // Combine shadows
    let combinedShadow: Float32Array = new Float32Array(bgW * bgH);
    for (let i = 0; i < bgW * bgH; i++) {
      combinedShadow[i] = Math.max(contactShadow[i], castShadowBlurred[i]);
    }

    // Apply depth warping if provided
    if (depthMap) {
      const depthData = this.extractAlphaMask(depthMap);
      const warped = this.applyDepthWarp(combinedShadow, depthData, bgW, bgH);
      // Copy warped values back
      for (let i = 0; i < bgW * bgH; i++) {
        combinedShadow[i] = warped[i];
      }
    }

    // Apply intensity and mask out foreground
    for (let i = 0; i < bgW * bgH; i++) {
      combinedShadow[i] = combinedShadow[i] * this.config.shadowIntensity * (1 - fullMask[i] * 0.95);
    }

    // Create shadow-only ImageData
    const shadowOnly = new ImageData(bgW, bgH);
    for (let i = 0; i < bgW * bgH; i++) {
      const val = Math.round(combinedShadow[i] * 255);
      shadowOnly.data[i * 4] = val;
      shadowOnly.data[i * 4 + 1] = val;
      shadowOnly.data[i * 4 + 2] = val;
      shadowOnly.data[i * 4 + 3] = 255;
    }

    // Create mask debug ImageData
    const maskDebug = new ImageData(bgW, bgH);
    for (let i = 0; i < bgW * bgH; i++) {
      const val = Math.round(fullMask[i] * 255);
      maskDebug.data[i * 4] = val;
      maskDebug.data[i * 4 + 1] = val;
      maskDebug.data[i * 4 + 2] = val;
      maskDebug.data[i * 4 + 3] = 255;
    }

    // Create composite
    const composite = new ImageData(bgW, bgH);
    for (let i = 0; i < bgW * bgH; i++) {
      const shadowFactor = 1 - combinedShadow[i] * 0.8;
      composite.data[i * 4] = Math.round(background.data[i * 4] * shadowFactor);
      composite.data[i * 4 + 1] = Math.round(background.data[i * 4 + 1] * shadowFactor);
      composite.data[i * 4 + 2] = Math.round(background.data[i * 4 + 2] * shadowFactor);
      composite.data[i * 4 + 3] = 255;
    }

    // Overlay foreground
    for (let y = 0; y < fgH; y++) {
      const destY = y + position.y;
      if (destY < 0 || destY >= bgH) continue;

      for (let x = 0; x < fgW; x++) {
        const destX = x + position.x;
        if (destX < 0 || destX >= bgW) continue;

        const srcIdx = (y * fgW + x) * 4;
        const destIdx = (destY * bgW + destX) * 4;
        const alpha = foreground.data[srcIdx + 3] / 255;

        if (alpha > 0) {
          composite.data[destIdx] = Math.round(
            foreground.data[srcIdx] * alpha + composite.data[destIdx] * (1 - alpha)
          );
          composite.data[destIdx + 1] = Math.round(
            foreground.data[srcIdx + 1] * alpha + composite.data[destIdx + 1] * (1 - alpha)
          );
          composite.data[destIdx + 2] = Math.round(
            foreground.data[srcIdx + 2] * alpha + composite.data[destIdx + 2] * (1 - alpha)
          );
        }
      }
    }

    return { composite, shadowOnly, maskDebug };
  }
}

/**
 * Scale an ImageData to a new size
 */
export function scaleImageData(imageData: ImageData, scale: number): ImageData {
  const newWidth = Math.round(imageData.width * scale);
  const newHeight = Math.round(imageData.height * scale);

  // Create temporary canvases for scaling
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = imageData.width;
  srcCanvas.height = imageData.height;
  const srcCtx = srcCanvas.getContext('2d')!;
  srcCtx.putImageData(imageData, 0, 0);

  const dstCanvas = document.createElement('canvas');
  dstCanvas.width = newWidth;
  dstCanvas.height = newHeight;
  const dstCtx = dstCanvas.getContext('2d')!;

  // Use high-quality scaling
  dstCtx.imageSmoothingEnabled = true;
  dstCtx.imageSmoothingQuality = 'high';
  dstCtx.drawImage(srcCanvas, 0, 0, newWidth, newHeight);

  return dstCtx.getImageData(0, 0, newWidth, newHeight);
}

/**
 * Remove background from an image (for JPG without transparency)
 * Detects gray/uniform backgrounds and removes them
 */
export function removeBackground(imageData: ImageData, tolerance: number = 50): ImageData {
  const { width, height, data } = imageData;
  const result = new ImageData(width, height);

  // Sample edge pixels to detect background color
  const edgePixels: number[][] = [];

  // Top edge
  for (let x = 0; x < width; x += 5) {
    for (let y = 0; y < Math.min(30, height); y++) {
      const idx = (y * width + x) * 4;
      edgePixels.push([data[idx], data[idx + 1], data[idx + 2]]);
    }
  }

  // Left and right edges
  for (let y = 0; y < height; y += 5) {
    for (let x = 0; x < Math.min(30, width); x++) {
      const idx = (y * width + x) * 4;
      edgePixels.push([data[idx], data[idx + 1], data[idx + 2]]);
    }
    for (let x = Math.max(0, width - 30); x < width; x++) {
      const idx = (y * width + x) * 4;
      edgePixels.push([data[idx], data[idx + 1], data[idx + 2]]);
    }
  }

  // Calculate median background color
  const sortedR = edgePixels.map(p => p[0]).sort((a, b) => a - b);
  const sortedG = edgePixels.map(p => p[1]).sort((a, b) => a - b);
  const sortedB = edgePixels.map(p => p[2]).sort((a, b) => a - b);
  const mid = Math.floor(edgePixels.length / 2);
  const bgColor = [sortedR[mid], sortedG[mid], sortedB[mid]];

  console.log(`Detected background color: RGB(${bgColor[0]}, ${bgColor[1]}, ${bgColor[2]})`);

  // Create alpha mask based on distance from background
  const alphaMask = new Float32Array(width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];

      // Distance from background color
      const dist = Math.sqrt(
        Math.pow(r - bgColor[0], 2) +
        Math.pow(g - bgColor[1], 2) +
        Math.pow(b - bgColor[2], 2)
      );

      // Pixels far from background get high alpha
      let alpha = Math.min(dist / tolerance, 1);

      // Make it more binary
      if (dist > tolerance * 0.7) alpha = 1;
      if (dist < tolerance * 0.3) alpha = 0;

      alphaMask[y * width + x] = alpha;
    }
  }

  // Simple morphological cleanup (dilate then erode)
  const cleaned = new Float32Array(width * height);
  const radius = 3;

  // Fill holes (dilate)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let maxVal = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const ny = Math.min(Math.max(y + dy, 0), height - 1);
          const nx = Math.min(Math.max(x + dx, 0), width - 1);
          maxVal = Math.max(maxVal, alphaMask[ny * width + nx]);
        }
      }
      cleaned[y * width + x] = maxVal;
    }
  }

  // Copy back with slight smoothing
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const alpha = cleaned[y * width + x];

      result.data[idx] = data[idx];
      result.data[idx + 1] = data[idx + 1];
      result.data[idx + 2] = data[idx + 2];
      result.data[idx + 3] = Math.round(alpha * 255);
    }
  }

  return result;
}

/**
 * Utility to load an image and get its ImageData
 */
export async function loadImage(src: string): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      resolve(ctx.getImageData(0, 0, img.width, img.height));
    };

    img.onerror = () => reject(new Error(`Failed to load image: ${src}`));
    img.src = src;
  });
}

/**
 * Utility to load from File object
 */
export async function loadImageFromFile(file: File): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = async (e) => {
      try {
        const data = await loadImage(e.target!.result as string);
        resolve(data);
      } catch (err) {
        reject(err);
      }
    };

    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

/**
 * Convert ImageData to a downloadable Blob
 */
export function imageDataToBlob(imageData: ImageData): Promise<Blob> {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const ctx = canvas.getContext('2d')!;
    ctx.putImageData(imageData, 0, 0);
    canvas.toBlob((blob) => resolve(blob!), 'image/png');
  });
}

/**
 * Display ImageData on a canvas element
 */
export function displayOnCanvas(imageData: ImageData, canvas: HTMLCanvasElement): void {
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  const ctx = canvas.getContext('2d')!;
  ctx.putImageData(imageData, 0, 0);
}
