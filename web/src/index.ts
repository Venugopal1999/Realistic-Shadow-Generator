/**
 * Realistic Shadow Generator - Web App Entry Point
 * Interactive UI for shadow generation with live preview
 */

import {
  ShadowGenerator,
  ShadowConfig,
  loadImageFromFile,
  displayOnCanvas,
  imageDataToBlob,
  removeBackground,
  scaleImageData
} from './shadow-generator';

class ShadowGeneratorApp {
  private generator: ShadowGenerator;

  // State
  private foregroundData: ImageData | null = null;
  private backgroundData: ImageData | null = null;
  private depthMapData: ImageData | null = null;

  // DOM Elements
  private foregroundInput!: HTMLInputElement;
  private backgroundInput!: HTMLInputElement;
  private depthMapInput!: HTMLInputElement;
  private angleSlider!: HTMLInputElement;
  private elevationSlider!: HTMLInputElement;
  private intensitySlider!: HTMLInputElement;
  private posXInput!: HTMLInputElement;
  private posYInput!: HTMLInputElement;
  private scaleSlider!: HTMLInputElement;
  private autoPositionCheckbox!: HTMLInputElement;
  private autoRemoveBgCheckbox!: HTMLInputElement;
  private generateBtn!: HTMLButtonElement;
  private downloadBtn!: HTMLButtonElement;

  // Preview canvases
  private compositeCanvas!: HTMLCanvasElement;
  private shadowCanvas!: HTMLCanvasElement;
  private maskCanvas!: HTMLCanvasElement;

  // Status
  private statusText!: HTMLElement;

  constructor() {
    this.generator = new ShadowGenerator();
    this.initializeDOM();
    this.bindEvents();
  }

  private initializeDOM(): void {
    // File inputs
    this.foregroundInput = document.getElementById('foreground-input') as HTMLInputElement;
    this.backgroundInput = document.getElementById('background-input') as HTMLInputElement;
    this.depthMapInput = document.getElementById('depth-map-input') as HTMLInputElement;

    // Sliders
    this.angleSlider = document.getElementById('angle-slider') as HTMLInputElement;
    this.elevationSlider = document.getElementById('elevation-slider') as HTMLInputElement;
    this.intensitySlider = document.getElementById('intensity-slider') as HTMLInputElement;

    // Position and scale inputs
    this.posXInput = document.getElementById('pos-x') as HTMLInputElement;
    this.posYInput = document.getElementById('pos-y') as HTMLInputElement;
    this.scaleSlider = document.getElementById('scale-slider') as HTMLInputElement;
    this.autoPositionCheckbox = document.getElementById('auto-position') as HTMLInputElement;
    this.autoRemoveBgCheckbox = document.getElementById('auto-remove-bg') as HTMLInputElement;

    // Buttons
    this.generateBtn = document.getElementById('generate-btn') as HTMLButtonElement;
    this.downloadBtn = document.getElementById('download-btn') as HTMLButtonElement;

    // Canvases
    this.compositeCanvas = document.getElementById('composite-canvas') as HTMLCanvasElement;
    this.shadowCanvas = document.getElementById('shadow-canvas') as HTMLCanvasElement;
    this.maskCanvas = document.getElementById('mask-canvas') as HTMLCanvasElement;

    // Status
    this.statusText = document.getElementById('status-text') as HTMLElement;
  }

  private bindEvents(): void {
    // File input handlers
    this.foregroundInput.addEventListener('change', (e) => this.handleFileInput(e, 'foreground'));
    this.backgroundInput.addEventListener('change', (e) => this.handleFileInput(e, 'background'));
    this.depthMapInput.addEventListener('change', (e) => this.handleFileInput(e, 'depth'));

    // Slider value display updates
    this.angleSlider.addEventListener('input', () => this.updateSliderDisplay());
    this.elevationSlider.addEventListener('input', () => this.updateSliderDisplay());
    this.intensitySlider.addEventListener('input', () => this.updateSliderDisplay());
    this.scaleSlider.addEventListener('input', () => this.updateSliderDisplay());

    // Auto-position toggle
    this.autoPositionCheckbox.addEventListener('change', () => this.updateAutoPosition());

    // Generate button
    this.generateBtn.addEventListener('click', () => this.generateShadow());

    // Download button
    this.downloadBtn.addEventListener('click', () => this.downloadResults());

    // Drag and drop on drop zones
    this.setupDropZone('foreground-drop', 'foreground');
    this.setupDropZone('background-drop', 'background');
    this.setupDropZone('depth-drop', 'depth');
  }

  private setupDropZone(elementId: string, type: 'foreground' | 'background' | 'depth'): void {
    const dropZone = document.getElementById(elementId);
    if (!dropZone) return;

    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
      dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', async (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');

      const files = e.dataTransfer?.files;
      if (files && files.length > 0) {
        await this.loadFile(files[0], type);
      }
    });

    dropZone.addEventListener('click', () => {
      const inputMap = {
        foreground: this.foregroundInput,
        background: this.backgroundInput,
        depth: this.depthMapInput
      };
      inputMap[type].click();
    });
  }

  private async handleFileInput(event: Event, type: 'foreground' | 'background' | 'depth'): Promise<void> {
    const input = event.target as HTMLInputElement;
    const files = input.files;
    if (files && files.length > 0) {
      await this.loadFile(files[0], type);
    }
  }

  private async loadFile(file: File, type: 'foreground' | 'background' | 'depth'): Promise<void> {
    this.setStatus(`Loading ${type}...`);
    console.log(`Loading ${type}:`, file.name, file.size, 'bytes');

    try {
      const imageData = await loadImageFromFile(file);
      console.log(`Loaded ${type}:`, imageData.width, 'x', imageData.height);

      switch (type) {
        case 'foreground':
          this.foregroundData = imageData;
          this.updateDropZonePreview('foreground-drop', imageData, file.name);
          break;
        case 'background':
          this.backgroundData = imageData;
          this.updateDropZonePreview('background-drop', imageData, file.name);
          break;
        case 'depth':
          this.depthMapData = imageData;
          this.updateDropZonePreview('depth-drop', imageData, file.name);
          break;
      }

      this.updateAutoPosition();
      this.setStatus(`${type} loaded: ${file.name}`);
    } catch (error) {
      this.setStatus(`Error loading ${type}: ${error}`, true);
    }
  }

  private updateDropZonePreview(elementId: string, imageData: ImageData, fileName: string): void {
    const dropZone = document.getElementById(elementId);
    if (!dropZone) return;

    // Create preview canvas
    const canvas = document.createElement('canvas');
    const maxSize = 100;
    const scale = Math.min(maxSize / imageData.width, maxSize / imageData.height);
    canvas.width = imageData.width * scale;
    canvas.height = imageData.height * scale;

    const ctx = canvas.getContext('2d')!;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    tempCanvas.getContext('2d')!.putImageData(imageData, 0, 0);

    ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);

    dropZone.innerHTML = '';
    dropZone.appendChild(canvas);

    const label = document.createElement('span');
    label.className = 'file-name';
    label.textContent = fileName.length > 20 ? fileName.substring(0, 17) + '...' : fileName;
    dropZone.appendChild(label);
  }

  private updateSliderDisplay(): void {
    const angleValue = document.getElementById('angle-value');
    const elevationValue = document.getElementById('elevation-value');
    const intensityValue = document.getElementById('intensity-value');
    const scaleValue = document.getElementById('scale-value');

    if (angleValue) angleValue.textContent = `${this.angleSlider.value}`;
    if (elevationValue) elevationValue.textContent = `${this.elevationSlider.value}`;
    if (intensityValue) intensityValue.textContent = (parseInt(this.intensitySlider.value) / 100).toFixed(2);
    if (scaleValue) scaleValue.textContent = `${this.scaleSlider.value}%`;
  }

  private updateAutoPosition(): void {
    const isAuto = this.autoPositionCheckbox.checked;
    this.posXInput.disabled = isAuto;
    this.posYInput.disabled = isAuto;

    if (isAuto && this.foregroundData && this.backgroundData) {
      const posX = Math.round((this.backgroundData.width - this.foregroundData.width) / 2);
      const posY = Math.round(this.backgroundData.height - this.foregroundData.height - 20);
      this.posXInput.value = String(posX);
      this.posYInput.value = String(posY);
    }
  }

  private async generateShadow(): Promise<void> {
    if (!this.foregroundData || !this.backgroundData) {
      this.setStatus('Please load both foreground and background images', true);
      return;
    }

    this.setStatus('Generating shadow...');
    this.generateBtn.disabled = true;

    // Use setTimeout to allow UI update
    await new Promise(resolve => setTimeout(resolve, 10));

    try {
      // Apply background removal if enabled
      let processedForeground = this.foregroundData;
      if (this.autoRemoveBgCheckbox && this.autoRemoveBgCheckbox.checked) {
        this.setStatus('Removing background...');
        await new Promise(resolve => setTimeout(resolve, 10));
        processedForeground = removeBackground(this.foregroundData);
        console.log('Background removed');
      }

      // Apply scaling
      const scale = parseInt(this.scaleSlider.value) / 100;
      if (scale !== 1) {
        this.setStatus(`Scaling to ${Math.round(scale * 100)}%...`);
        await new Promise(resolve => setTimeout(resolve, 10));
        processedForeground = scaleImageData(processedForeground, scale);
        console.log(`Scaled to ${processedForeground.width}x${processedForeground.height}`);
      }

      this.setStatus('Generating shadow...');
      await new Promise(resolve => setTimeout(resolve, 10));

      // Update generator config
      this.generator.updateConfig({
        lightAngle: parseInt(this.angleSlider.value),
        lightElevation: parseInt(this.elevationSlider.value),
        shadowIntensity: parseInt(this.intensitySlider.value) / 100
      });

      const position = {
        x: parseInt(this.posXInput.value),
        y: parseInt(this.posYInput.value)
      };

      const result = this.generator.generate(
        processedForeground,
        this.backgroundData,
        this.depthMapData,
        position
      );

      // Display results
      displayOnCanvas(result.composite, this.compositeCanvas);
      displayOnCanvas(result.shadowOnly, this.shadowCanvas);
      displayOnCanvas(result.maskDebug, this.maskCanvas);

      this.downloadBtn.disabled = false;
      this.setStatus('Shadow generated successfully!');
    } catch (error) {
      this.setStatus(`Error generating shadow: ${error}`, true);
    } finally {
      this.generateBtn.disabled = false;
    }
  }

  private async downloadResults(): Promise<void> {
    const compositeCtx = this.compositeCanvas.getContext('2d');
    const shadowCtx = this.shadowCanvas.getContext('2d');
    const maskCtx = this.maskCanvas.getContext('2d');

    if (!compositeCtx || !shadowCtx || !maskCtx) return;

    this.setStatus('Preparing downloads...');

    try {
      // Download all three images
      await this.downloadCanvas(this.compositeCanvas, 'composite.png');
      await this.downloadCanvas(this.shadowCanvas, 'shadow_only.png');
      await this.downloadCanvas(this.maskCanvas, 'mask_debug.png');

      this.setStatus('All files downloaded!');
    } catch (error) {
      this.setStatus(`Download error: ${error}`, true);
    }
  }

  private async downloadCanvas(canvas: HTMLCanvasElement, filename: string): Promise<void> {
    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        if (blob) {
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = filename;
          a.click();
          URL.revokeObjectURL(url);
        }
        resolve();
      }, 'image/png');
    });
  }

  private setStatus(message: string, isError: boolean = false): void {
    this.statusText.textContent = message;
    this.statusText.className = isError ? 'status-error' : 'status-success';
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  console.log('Shadow Generator: Initializing...');
  try {
    const app = new ShadowGeneratorApp();
    console.log('Shadow Generator: Ready!');
    (window as any).shadowApp = app; // For debugging
  } catch (error) {
    console.error('Shadow Generator: Failed to initialize', error);
  }
});
