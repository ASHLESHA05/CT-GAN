import sys
import os
import numpy as np
import time

import os
import sys
# Store command-line arguments or default to "b"
os.environ["CLI_ARGS"] = " ".join(arg.lower() for arg in sys.argv[1:]) if len(sys.argv) > 1 else "b"
# Retrieve the stored arguments
cli_args = os.environ.get("CLI_ARGS", "").split()

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QLabel, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from procedures.attack_pipeline import scan_manipulator
from utils.equalizer import histEq

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('CT-GAN: 3D Medical Scan Manipulation')
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")
        
        self.manipulator = None
        self.filepaths = []
        self.fileindex = 0
        self.hist_state = True
        self.inject_coords = []
        self.remove_coords = []
        self.animation_running = False

        self.initUI()
    
    def initUI(self):
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.button_layout = QHBoxLayout()
        self.top_button_layout = QHBoxLayout()

        self.fig, self.ax = plt.subplots(1, 1, dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout.addWidget(self.toolbar)
        self.main_layout.addWidget(self.canvas)

        # Buttons
        self.btn_load = QPushButton('Load Scans')
        self.btn_refresh = QPushButton('Refresh')
        self.btn_toggle_animation = QPushButton('Start Animation')
        self.btn_hist = QPushButton('Toggle HistEQ')
        self.btn_prev = QPushButton('Previous')
        self.btn_save = QPushButton('Save')
        self.btn_next = QPushButton('Next')

        self.btn_inject = QPushButton('Inject')
        self.btn_remove = QPushButton('Remove')
        
        self.style_buttons([self.btn_load, self.btn_refresh, self.btn_toggle_animation, self.btn_hist,
                            self.btn_prev, self.btn_save, self.btn_next, self.btn_inject, self.btn_remove])

        self.top_button_layout.addWidget(self.btn_load)
        self.top_button_layout.addWidget(self.btn_refresh)
        self.top_button_layout.addWidget(self.btn_toggle_animation)
        
        self.button_layout.addWidget(self.btn_inject)
        self.button_layout.addWidget(self.btn_hist)
        self.button_layout.addWidget(self.btn_prev)
        self.button_layout.addWidget(self.btn_save)
        self.button_layout.addWidget(self.btn_next)
        self.button_layout.addWidget(self.btn_remove)
        
        self.main_layout.addLayout(self.top_button_layout)
        self.main_layout.addLayout(self.button_layout)
        
        self.btn_load.clicked.connect(self.load_scans)
        self.btn_refresh.clicked.connect(self.refresh_scan)
        self.btn_toggle_animation.clicked.connect(self.toggle_animation)
        self.btn_inject.clicked.connect(lambda: self.set_action_state('inject'))
        self.btn_remove.clicked.connect(lambda: self.set_action_state('remove'))
        self.btn_hist.clicked.connect(self.hist)
        self.btn_prev.clicked.connect(self.prev)
        self.btn_save.clicked.connect(self.save)
        self.btn_next.clicked.connect(self.next)
        
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.mpl_connect('scroll_event', self.scroll)
        
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate)
    
    def style_buttons(self, buttons):
        for btn in buttons:
            btn.setStyleSheet("background-color: #444; border-radius: 10px; padding: 10px;")
            btn.setFont(QFont('Arial', 10))
    
    def load_scans(self):
        path = QFileDialog.getExistingDirectory(self, "Select Scan Folder")
        if not path:
            return
        
        self.filepaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.dcm') or f.endswith('.mhd')]
        if not self.filepaths:
            QMessageBox.critical(self, "Error", "No valid scan files found!")
            return
        
        self.fileindex = 0
        self.load_scan()
    
    def refresh_scan(self):
        if self.filepaths:
            self.load_scan()
    
    def load_scan(self):
        self.manipulator = scan_manipulator()
        self.manipulator.load_target_scan(self.filepaths[self.fileindex])
        self.eq = histEq(self.manipulator.scan)
        self.slices, self.cols, self.rows = self.manipulator.scan.shape
        self.ind = self.slices // 2
        self.plot()
    
    def toggle_animation(self):
        if self.animation_running:
            self.animation_timer.stop()
            self.btn_toggle_animation.setText("Start Animation")
        else:
            self.animation_timer.start(100)
            self.btn_toggle_animation.setText("Stop Animation")
        self.animation_running = not self.animation_running
    
    def animate(self):
        self.ind = (self.ind + 1) % self.slices
        self.update_plot()
    
    def onclick(self, event):
        if event.xdata is None or event.ydata is None:
            return
        coord = np.array([self.ind, int(event.ydata), int(event.xdata)], dtype=int)
        if self.action_state == 'inject':
            self.manipulator.tamper(coord, action='inject', isVox=True)
            self.inject_coords.append(coord)
        elif self.action_state == 'remove':
            self.manipulator.tamper(coord, action='remove', isVox=True)
            self.remove_coords.append(coord)
        self.update_plot()
    
    def set_action_state(self, state):
        self.action_state = state
    
    def hist(self):
        self.hist_state = not self.hist_state
        self.update_plot()
    
    def prev(self):
        self.fileindex = (self.fileindex - 1) % len(self.filepaths)
        self.load_scan()
    
    def next(self):
        self.fileindex = (self.fileindex + 1) % len(self.filepaths)
        self.load_scan()
    
    def save(self):
        if not self.manipulator:
            return
        self.manipulator.save_tampered_scan("output2.dcm", output_type='dicom')
    
    def plot(self):
        """Initial plot function."""
        self.ax.clear()
        image_data = self.eq.equalize(self.manipulator.scan[self.ind, :, :]) if self.hist_state else self.manipulator.scan[self.ind, :, :]
        self.im = self.ax.imshow(image_data, cmap="bone")
        self.canvas.draw()

    def update_plot(self):
        """Handles toggling histogram equalization correctly."""
        self.ax.clear()  # Clear the previous plot completely
        image_data = self.eq.equalize(self.manipulator.scan[self.ind, :, :]) if self.hist_state else self.manipulator.scan[self.ind, :, :]
        
        # Replot the image properly with updated data
        self.im = self.ax.imshow(image_data, cmap="bone")
        
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
