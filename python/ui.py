#!/usr/bin/env python3
"""
Realistic Shadow Generator - PyQt6 GUI
Interactive UI for shadow generation with live preview.
"""

import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QGroupBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QProgressBar, QSplitter, QFrame,
    QScrollArea, QSizePolicy, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QFont

from PIL import Image
import numpy as np

from shadow_generator import ShadowGenerator, auto_cutout_subject


class ImageLabel(QLabel):
    """Custom label for displaying images with drag-drop support."""

    def __init__(self, placeholder_text: str = "Drop image here"):
        super().__init__()
        self.placeholder_text = placeholder_text
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self.setAcceptDrops(True)
        self.setText(placeholder_text)
        self.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                border: 2px dashed #555;
                border-radius: 8px;
                color: #888;
                font-size: 14px;
            }
        """)
        self._image_path = None

    def set_image(self, path: str):
        """Load and display image from path."""
        self._image_path = path
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled)

    def set_pixmap_from_pil(self, pil_image: Image.Image):
        """Display PIL image."""
        if pil_image.mode == 'RGBA':
            data = pil_image.tobytes("raw", "RGBA")
            qimg = QImage(data, pil_image.width, pil_image.height, QImage.Format.Format_RGBA8888)
        elif pil_image.mode == 'RGB':
            data = pil_image.tobytes("raw", "RGB")
            qimg = QImage(data, pil_image.width, pil_image.height, QImage.Format.Format_RGB888)
        else:
            pil_image = pil_image.convert('RGB')
            data = pil_image.tobytes("raw", "RGB")
            qimg = QImage(data, pil_image.width, pil_image.height, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)

    @property
    def image_path(self):
        return self._image_path

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.set_image(path)


class ShadowWorker(QThread):
    """Background worker for shadow generation."""

    finished = pyqtSignal(object, object, object)  # composite, shadow, mask
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(
        self,
        foreground: Image.Image,
        background: Image.Image,
        depth_map: Optional[Image.Image],
        position: tuple,
        light_angle: float,
        light_elevation: float,
        intensity: float
    ):
        super().__init__()
        self.foreground = foreground
        self.background = background
        self.depth_map = depth_map
        self.position = position
        self.light_angle = light_angle
        self.light_elevation = light_elevation
        self.intensity = intensity

    def run(self):
        try:
            self.progress.emit(20)

            generator = ShadowGenerator(
                light_angle=self.light_angle,
                light_elevation=self.light_elevation,
                shadow_intensity=self.intensity
            )

            self.progress.emit(50)

            composite, shadow_only, mask_debug = generator.generate_shadow(
                self.foreground,
                self.background,
                self.depth_map,
                self.position
            )

            self.progress.emit(100)
            self.finished.emit(composite, shadow_only, mask_debug)

        except Exception as e:
            self.error.emit(str(e))


class ShadowGeneratorUI(QMainWindow):
    """Main window for Shadow Generator application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realistic Shadow Generator")
        self.setMinimumSize(1200, 800)

        # State
        self.foreground_image: Optional[Image.Image] = None
        self.background_image: Optional[Image.Image] = None
        self.depth_map: Optional[Image.Image] = None
        self.composite_result: Optional[Image.Image] = None
        self.shadow_result: Optional[Image.Image] = None
        self.mask_result: Optional[Image.Image] = None

        self.setup_ui()
        self.apply_dark_theme()

    def setup_ui(self):
        """Setup the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        # Left panel - controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)

        # Right panel - preview
        right_panel = self.create_preview_panel()
        main_layout.addWidget(right_panel, 2)

    def create_control_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Input images group
        input_group = QGroupBox("Input Images")
        input_layout = QVBoxLayout(input_group)

        # Foreground
        fg_layout = QHBoxLayout()
        fg_layout.addWidget(QLabel("Foreground:"))
        self.fg_btn = QPushButton("Browse...")
        self.fg_btn.clicked.connect(self.load_foreground)
        fg_layout.addWidget(self.fg_btn)
        self.fg_label = QLabel("No file selected")
        self.fg_label.setStyleSheet("color: #888;")
        input_layout.addLayout(fg_layout)
        input_layout.addWidget(self.fg_label)

        # Auto cutout checkbox
        self.auto_cutout_cb = QCheckBox("Auto-cutout foreground subject")
        input_layout.addWidget(self.auto_cutout_cb)

        # Background
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel("Background:"))
        self.bg_btn = QPushButton("Browse...")
        self.bg_btn.clicked.connect(self.load_background)
        bg_layout.addWidget(self.bg_btn)
        self.bg_label = QLabel("No file selected")
        self.bg_label.setStyleSheet("color: #888;")
        input_layout.addLayout(bg_layout)
        input_layout.addWidget(self.bg_label)

        # Depth map (optional)
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Depth Map:"))
        self.depth_btn = QPushButton("Browse...")
        self.depth_btn.clicked.connect(self.load_depth_map)
        depth_layout.addWidget(self.depth_btn)
        self.depth_label = QLabel("Optional")
        self.depth_label.setStyleSheet("color: #666;")
        input_layout.addLayout(depth_layout)
        input_layout.addWidget(self.depth_label)

        layout.addWidget(input_group)

        # Light settings group
        light_group = QGroupBox("Light Settings")
        light_layout = QVBoxLayout(light_group)

        # Light angle
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Angle (0-360):"))
        self.angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.angle_slider.setRange(0, 360)
        self.angle_slider.setValue(135)
        self.angle_slider.valueChanged.connect(self.on_settings_changed)
        angle_layout.addWidget(self.angle_slider)
        self.angle_value = QLabel("135")
        self.angle_value.setMinimumWidth(40)
        angle_layout.addWidget(self.angle_value)
        light_layout.addLayout(angle_layout)

        # Light elevation
        elev_layout = QHBoxLayout()
        elev_layout.addWidget(QLabel("Elevation (0-90):"))
        self.elev_slider = QSlider(Qt.Orientation.Horizontal)
        self.elev_slider.setRange(0, 90)
        self.elev_slider.setValue(45)
        self.elev_slider.valueChanged.connect(self.on_settings_changed)
        elev_layout.addWidget(self.elev_slider)
        self.elev_value = QLabel("45")
        self.elev_value.setMinimumWidth(40)
        elev_layout.addWidget(self.elev_value)
        light_layout.addLayout(elev_layout)

        # Shadow intensity
        int_layout = QHBoxLayout()
        int_layout.addWidget(QLabel("Intensity:"))
        self.int_slider = QSlider(Qt.Orientation.Horizontal)
        self.int_slider.setRange(0, 100)
        self.int_slider.setValue(70)
        self.int_slider.valueChanged.connect(self.on_settings_changed)
        int_layout.addWidget(self.int_slider)
        self.int_value = QLabel("0.7")
        self.int_value.setMinimumWidth(40)
        int_layout.addWidget(self.int_value)
        light_layout.addLayout(int_layout)

        layout.addWidget(light_group)

        # Position group
        pos_group = QGroupBox("Foreground Position")
        pos_layout = QVBoxLayout(pos_group)

        pos_x_layout = QHBoxLayout()
        pos_x_layout.addWidget(QLabel("X:"))
        self.pos_x_spin = QSpinBox()
        self.pos_x_spin.setRange(-2000, 2000)
        self.pos_x_spin.setValue(0)
        self.pos_x_spin.valueChanged.connect(self.on_settings_changed)
        pos_x_layout.addWidget(self.pos_x_spin)
        pos_layout.addLayout(pos_x_layout)

        pos_y_layout = QHBoxLayout()
        pos_y_layout.addWidget(QLabel("Y:"))
        self.pos_y_spin = QSpinBox()
        self.pos_y_spin.setRange(-2000, 2000)
        self.pos_y_spin.setValue(0)
        self.pos_y_spin.valueChanged.connect(self.on_settings_changed)
        pos_y_layout.addWidget(self.pos_y_spin)
        pos_layout.addLayout(pos_y_layout)

        self.auto_pos_cb = QCheckBox("Auto-position (center bottom)")
        self.auto_pos_cb.setChecked(True)
        self.auto_pos_cb.stateChanged.connect(self.on_auto_pos_changed)
        pos_layout.addWidget(self.auto_pos_cb)

        layout.addWidget(pos_group)

        # Action buttons
        action_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Shadow")
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #0066cc;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0077ee;
            }
            QPushButton:disabled {
                background-color: #444;
            }
        """)
        self.generate_btn.clicked.connect(self.generate_shadow)
        action_layout.addWidget(self.generate_btn)
        layout.addLayout(action_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Save button
        self.save_btn = QPushButton("Save Results")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_results)
        layout.addWidget(self.save_btn)

        layout.addStretch()

        return panel

    def create_preview_panel(self) -> QWidget:
        """Create the right preview panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Preview tabs/sections
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)

        # Main composite preview
        self.preview_label = ImageLabel("Preview will appear here")
        self.preview_label.setMinimumSize(400, 300)
        preview_layout.addWidget(self.preview_label)

        layout.addWidget(preview_group, 2)

        # Debug previews
        debug_group = QGroupBox("Debug Views")
        debug_layout = QHBoxLayout(debug_group)

        # Shadow only
        shadow_frame = QVBoxLayout()
        shadow_frame.addWidget(QLabel("Shadow Only"))
        self.shadow_preview = ImageLabel("Shadow")
        self.shadow_preview.setMinimumSize(150, 100)
        shadow_frame.addWidget(self.shadow_preview)
        debug_layout.addLayout(shadow_frame)

        # Mask debug
        mask_frame = QVBoxLayout()
        mask_frame.addWidget(QLabel("Mask Debug"))
        self.mask_preview = ImageLabel("Mask")
        self.mask_preview.setMinimumSize(150, 100)
        mask_frame.addWidget(self.mask_preview)
        debug_layout.addLayout(mask_frame)

        layout.addWidget(debug_group, 1)

        return panel

    def apply_dark_theme(self):
        """Apply dark theme to the application."""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3d3d3d;
                border: 1px solid #555;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #3d3d3d;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0066cc;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #2d2d2d;
                border: 1px solid #444;
                padding: 3px;
                border-radius: 3px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0066cc;
            }
        """)

    def load_foreground(self):
        """Load foreground image."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Foreground Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if path:
            self.foreground_image = Image.open(path)
            self.fg_label.setText(os.path.basename(path))
            self.update_auto_position()

    def load_background(self):
        """Load background image."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Background Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if path:
            self.background_image = Image.open(path)
            self.bg_label.setText(os.path.basename(path))
            self.update_auto_position()

    def load_depth_map(self):
        """Load depth map image."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Depth Map",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.depth_map = Image.open(path)
            self.depth_label.setText(os.path.basename(path))

    def update_auto_position(self):
        """Update auto position based on loaded images."""
        if self.auto_pos_cb.isChecked() and self.foreground_image and self.background_image:
            bg_w, bg_h = self.background_image.size
            fg_w, fg_h = self.foreground_image.size
            self.pos_x_spin.setValue((bg_w - fg_w) // 2)
            self.pos_y_spin.setValue(bg_h - fg_h - 20)

    def on_auto_pos_changed(self, state):
        """Handle auto-position checkbox change."""
        enabled = state != Qt.CheckState.Checked.value
        self.pos_x_spin.setEnabled(enabled)
        self.pos_y_spin.setEnabled(enabled)
        if not enabled:
            self.update_auto_position()

    def on_settings_changed(self):
        """Handle slider/spinbox value changes."""
        self.angle_value.setText(str(self.angle_slider.value()))
        self.elev_value.setText(str(self.elev_slider.value()))
        self.int_value.setText(f"{self.int_slider.value() / 100:.2f}")

    def generate_shadow(self):
        """Generate shadow composite."""
        if not self.foreground_image or not self.background_image:
            QMessageBox.warning(
                self,
                "Missing Input",
                "Please load both foreground and background images."
            )
            return

        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Apply auto-cutout if needed
        fg = self.foreground_image
        if self.auto_cutout_cb.isChecked():
            try:
                fg = auto_cutout_subject(fg)
            except Exception as e:
                QMessageBox.warning(self, "Cutout Failed", f"Auto-cutout failed: {e}")

        # Get position
        position = (self.pos_x_spin.value(), self.pos_y_spin.value())

        # Start worker thread
        self.worker = ShadowWorker(
            foreground=fg,
            background=self.background_image.copy(),
            depth_map=self.depth_map,
            position=position,
            light_angle=self.angle_slider.value(),
            light_elevation=self.elev_slider.value(),
            intensity=self.int_slider.value() / 100.0
        )

        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_generation_complete)
        self.worker.error.connect(self.on_generation_error)
        self.worker.start()

    def on_progress(self, value):
        """Update progress bar."""
        self.progress_bar.setValue(value)

    def on_generation_complete(self, composite, shadow, mask):
        """Handle successful generation."""
        self.composite_result = composite
        self.shadow_result = shadow
        self.mask_result = mask

        # Update previews
        self.preview_label.set_pixmap_from_pil(composite)
        self.shadow_preview.set_pixmap_from_pil(shadow)
        self.mask_preview.set_pixmap_from_pil(mask)

        self.generate_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def on_generation_error(self, error_msg):
        """Handle generation error."""
        QMessageBox.critical(self, "Error", f"Shadow generation failed: {error_msg}")
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def save_results(self):
        """Save generated results."""
        if not self.composite_result:
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            try:
                self.composite_result.save(os.path.join(folder, "composite.png"))
                self.shadow_result.save(os.path.join(folder, "shadow_only.png"))
                self.mask_result.save(os.path.join(folder, "mask_debug.png"))
                QMessageBox.information(self, "Saved", f"Results saved to {folder}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = ShadowGeneratorUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
