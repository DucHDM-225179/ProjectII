import sys
from PyQt5.QtCore import Qt, QPointF, QTimer
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsItem,
    QAction, QActionGroup, QToolBar, QDockWidget, QWidget,
    QFormLayout, QDoubleSpinBox, QSpinBox, QPushButton,
    QVBoxLayout, QHBoxLayout, QDialog, QDialogButtonBox,
    QFileDialog
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import json
from riskaverseqlearning import RiskAverseQLearning
from deepqlearning_simple import DeepQLearning
from deepqlearning_riskaverse import RiskAverseDeepQLearning
import random

import os
from datetime import datetime

# ---- Constants ----
AP_COORD = QPointF(0, 0)
AP_RADIUS = 5
POINT_SIZE = 5
PACKET_SIZE = 8000
FRAME_DURATION = 1e-3
USE_RISK_AVERSE = False
USE_DEEP_RISK = True

SUFFIX_STR = ""
if USE_RISK_AVERSE:
    SUFFIX_STR = "RA"
else:
    if USE_DEEP_RISK:
        SUFFIX_STR = "DEEPRA"
    else:
        SUFFIX_STR = "DEEP"

# ---- Graph Window ----
class GraphWindow(QMainWindow):
    def __init__(self, title="Graph", xlabel="X", ylabel="Y", fit_to_screen=True, padding=(0.1, 0.1)):
        super().__init__()
        self.setWindowTitle(title)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fit_to_screen = fit_to_screen
        self.padding = padding
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.lines = {}

    def add_series(self, name, x=None, y=None):
        if name in self.lines:
            return
        line, = self.ax.plot(x or [], y or [], label=name)
        self.lines[name] = {'line': line, 'x': list(x or []), 'y': list(y or [])}
        self.ax.legend()
        self._rescale()
        self.canvas.draw()

    def add_point(self, name, x, y):
        if name not in self.lines:
            self.add_series(name, [x], [y])
            return
        series = self.lines[name]
        series['x'].append(x)
        series['y'].append(y)
        series['line'].set_data(series['x'], series['y'])
        self._rescale()
        self.canvas.draw()

    def clear(self):
        self.ax.cla()
        self.lines.clear()
        self.ax.set_title(self.windowTitle())
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.canvas.draw()

    def _rescale(self):
        all_x = []
        all_y = []
        for s in self.lines.values():
            all_x.extend(s['x'])
            all_y.extend(s['y'])
        if not all_x or not all_y:
            return
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        if self.fit_to_screen:
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
        else:
            dx = (xmax - xmin) * self.padding[0]
            dy = (ymax - ymin) * self.padding[1]
            self.ax.set_xlim(xmin - dx, xmax + dx)
            self.ax.set_ylim(ymin - dy, ymax + dy)

# ---- Graphics Items ----

class BaseItemMixin:
    def __init__(self, window):
        super().__init__()
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.position = (0, 0)
        self.window = window

    def round_to_SIZE(self, x):
        return POINT_SIZE * round(x/POINT_SIZE)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            new_pos = value
            self.position = ( self.round_to_SIZE(new_pos.x()), self.round_to_SIZE(new_pos.y()) )
            return QPointF(self.position[0], self.position[1])
        return QGraphicsItem.itemChange(self, change, value)

    def hoverEnterEvent(self, event):
        self.setOpacity(0.7)
        QGraphicsItem.hoverEnterEvent(self, event)

    def hoverLeaveEvent(self, event):
        self.setOpacity(1.0)
        QGraphicsItem.hoverLeaveEvent(self, event)

class DeviceItem(QGraphicsEllipseItem, BaseItemMixin):
    def __init__(self, pos, window):
        super().__init__(-POINT_SIZE / 2, -POINT_SIZE / 2, POINT_SIZE, POINT_SIZE, window=window)
        BaseItemMixin.__init__(self, window)
        self.setBrush(QBrush(QColor("blue")))
        self.setPos(pos)
        self.sub6_packets = 0
        self.mmwave_packets = 0
        self.history = {"sub6_success": [], "mmwave_success": []}

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            dlg = DeviceConfigDialog(self)
            dlg.exec_()
        else:
            super().mouseDoubleClickEvent(event)

class BlockageItem(QGraphicsRectItem, BaseItemMixin):
    def __init__(self, pos, window):
        super().__init__(-POINT_SIZE / 2, -POINT_SIZE / 2, POINT_SIZE, POINT_SIZE, window=window)
        BaseItemMixin.__init__(self, window)
        self.setBrush(QBrush(QColor("gray")))
        self.setPos(pos)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            dlg = BlockageConfigDialog(self)
            dlg.exec_()
        else:
            super().mouseDoubleClickEvent(event)

# ---- Config Dialogs ----

class DeviceConfigDialog(QDialog):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.setWindowTitle("Configure Device")
        vlay = QVBoxLayout(self)
        form = QFormLayout()

        self.sub6_spin = QSpinBox()
        self.sub6_spin.setRange(0, 10**6)
        self.sub6_spin.setValue(device.sub6_packets)
        form.addRow("Sub-6GHz packets:", self.sub6_spin)

        self.mm_spin = QSpinBox()
        self.mm_spin.setRange(0, 10**6)
        self.mm_spin.setValue(device.mmwave_packets)
        form.addRow("mmWave packets:", self.mm_spin)

        vlay.addLayout(form)
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save)
        
        del_btn = QPushButton("Delete")
        del_btn.clicked.connect(self.delete)
        
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(del_btn)
        
        vlay.addLayout(btn_layout)

    def save(self):
        self.device.sub6_packets = self.sub6_spin.value()
        self.device.mmwave_packets = self.mm_spin.value()
        self.accept()

    def delete(self):
        if self.device.window:
            self.device.window.removeItem(self.device)
        scene = self.device.scene()
        if scene:
            scene.removeItem(self.device)
        self.accept()

class BlockageConfigDialog(QDialog):
    def __init__(self, blockage):
        super().__init__()
        self.blockage = blockage
        self.setWindowTitle("Blockage Options")
        vlay = QVBoxLayout(self)
        del_btn = QPushButton("Delete Blockage")
        del_btn.clicked.connect(self.delete)
        vlay.addWidget(del_btn)
        close_btn = QDialogButtonBox(QDialogButtonBox.Close)
        close_btn.rejected.connect(self.reject)
        vlay.addWidget(close_btn)

    def delete(self):
        if self.blockage.window:
            self.blockage.window.removeItem(self.blockage)
        scene = self.blockage.scene()
        if scene:
            scene.removeItem(self.blockage)
        self.accept()

# ---- Global Config Widget ----

class GlobalConfigWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QFormLayout(self)

        self.pwr = QDoubleSpinBox()
        self.pwr.setMinimum(1)
        self.pwr.setMaximum(10)
        self.pwr.setValue(5)

        self.noise = QDoubleSpinBox()
        self.noise.setMinimum(-1000)
        self.noise.setMaximum(-1)
        self.noise.setValue(-169)

        self.bw_sub6 = QDoubleSpinBox()
        self.bw_sub6.setMinimum(1)
        self.bw_sub6.setMaximum(10000)
        self.bw_sub6.setValue(100)

        self.bw_mm = QDoubleSpinBox()
        self.bw_mm.setMinimum(1)
        self.bw_mm.setMaximum(10000)
        self.bw_mm.setValue(1000)
        
        #self.freq_sub6 = QDoubleSpinBox()
        #self.freq_sub6.setMinimum(1)
        #self.freq_sub6.setMaximum(100)
        #self.freq_sub6.setValue(2)
        
        #self.freq_mm = QDoubleSpinBox()  
        #self.freq_mm.setMinimum(1)
        #self.freq_mm.setMaximum(100)
        #self.freq_mm.setValue(28)
        
        self.num_subchannel = QSpinBox()
        self.num_subchannel.setMinimum(1)
        self.num_subchannel.setMaximum(100)
        self.num_subchannel.setValue(4)
        
        self.num_beam = QSpinBox()
        self.num_beam.setMinimum(1)
        self.num_beam.setMaximum(100)
        self.num_beam.setValue(4)
        
        #self.frame_duration = QDoubleSpinBox()  
        #self.frame_duration.setMinimum(1)
        #self.frame_duration.setMaximum(100000)
        #self.frame_duration.setValue(1000)
        
        self.nframes = QSpinBox()        
        self.nframes.setMinimum(1)
        self.nframes.setMaximum(100000)
        self.nframes.setValue(10000)
        
        for label, widget in [
            ("Tx power (dBm):", self.pwr),
            ("Noise power (dBm):", self.noise),
            ("BW Sub-6 (MHz):", self.bw_sub6),
            ("BW mmWave (MHz):", self.bw_mm),
            ("#Sub channels:", self.num_subchannel),
            ("#Sub beams:", self.num_beam),
            #("Freq Sub-6 (GHz):", self.freq_sub6),
            #("Freq mmWave (GHz):", self.freq_mm),
            #("Frame duration: ", self.frame_duration),
            ("# Frames to simulate:", self.nframes),
        ]:
            layout.addRow(label, widget)

# ---- Graphics View with Zoom & Pan ----



class GridView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing)
        self._pan = False
        self._last_pos = None
        self.setDragMode(self.RubberBandDrag)

    def wheelEvent(self, e):
        factor = 1.2 if e.angleDelta().y() > 0 else 1/1.2
        self.scale(factor, factor)

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self._pan = True
            self._last_pos = e.pos()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._pan and self._last_pos is not None:
            delta = e.pos() - self._last_pos
            self._last_pos = e.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        else:
            super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.RightButton:
            self._pan = False
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(e)

# ---- Main Window ----

class GridScene(QGraphicsScene):
    def __init__(self, x, y, width, height, grid_step, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSceneRect(x, y, width, height)
        self.grid_step = grid_step
        self.draw_grid()

    def draw_grid(self):
        pen = QPen(QColor(200, 200, 200), 1, Qt.SolidLine)
        left = int(self.sceneRect().left())
        right = int(self.sceneRect().right())
        top = int(self.sceneRect().top())
        bottom = int(self.sceneRect().bottom())

        # Vertical lines
        for x in range(left, right + 1, self.grid_step):
            self.addLine(x, top, x, bottom, pen)

        # Horizontal lines
        for y in range(top, bottom + 1, self.grid_step):
            self.addLine(left, y, right, y, pen)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._create_menu()
        self.setWindowTitle("5G Network Simulator")
        self.scene = GridScene(-500, -500, 1000, 1000, POINT_SIZE)
        self.view = GridView(self.scene)
        self.setCentralWidget(self.view)
        
        self.graph_windows = {}
        # Pre-create any hardcoded graphs
        for g in [("Metric","Frame","ΔP"), ("Reward", "Frame", "Reward")]:
            name,xl,yl=g
            gw=GraphWindow(title=name, xlabel=xl, ylabel=yl, fit_to_screen=False)
            gw.show()
            self.graph_windows[name]=gw

        ap = QGraphicsEllipseItem(-AP_RADIUS, -AP_RADIUS, AP_RADIUS*2, AP_RADIUS*2)
        ap.setBrush(QBrush(QColor("red")))
        ap.setPos(AP_COORD)
        self.scene.addItem(ap)
        
        toolbar = QToolBar("Tools")
        self.addToolBar(toolbar)
        
        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)
        
        self.dev_act = QAction("Place Device", self, checkable=True)
        self.dev_act.setChecked(True)
        
        self.blk_act = QAction("Place Blockage", self, checkable=True)
        
        mode_group.addAction(self.dev_act)
        mode_group.addAction(self.blk_act)
        
        toolbar.addAction(self.dev_act)
        toolbar.addAction(self.blk_act)
        
        self.dev_act.triggered.connect(lambda: setattr(self, 'mode', 'device'))
        self.blk_act.triggered.connect(lambda: setattr(self, 'mode', 'blockage'))
        
        self.mode = 'device'
        self.scene.mousePressEvent = self._scene_click

        cfg_dock = QDockWidget("Global Config", self)
        cfg_widget = GlobalConfigWidget()
        cfg_dock.setWidget(cfg_widget)
        self.cfg = cfg_widget
        self.addDockWidget(Qt.RightDockWidgetArea, cfg_dock)

        ctrl_dock = QDockWidget("Simulation Controls", self)
        box = QWidget()
        vlay = QVBoxLayout(box)
        self.start_btn = QPushButton("Start Simulation")
        vlay.addWidget(self.start_btn)
        
        self.step_btn = QPushButton("Step One")
        vlay.addWidget(self.step_btn)
        
        self.auto_btn = QPushButton("Auto Step")
        vlay.addWidget(self.auto_btn)
        
        self.reset_btn = QPushButton("Reset Simulation")
        vlay.addWidget(self.reset_btn)
        
        self.step_btn.setEnabled(False)
        self.auto_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        self.step_btn.clicked.connect(self._on_step)
        self.reset_btn.clicked.connect(self._on_reset)
        
        self.auto_btn.clicked.connect(self._on_auto)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_step)
        ctrl_dock.setWidget(box)
        self.addDockWidget(Qt.RightDockWidgetArea, ctrl_dock)
        self.devices = []
        self.blockages = []

    def _create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        save_action = QAction("Save JSON", self)
        save_action.triggered.connect(self.save_json)
        file_menu.addAction(save_action)

        load_action = QAction("Load JSON", self)
        load_action.triggered.connect(self.load_json)
        file_menu.addAction(load_action)

    def save_json(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Layout + Settings", "", "JSON Files (*.json)"
        )
        if not path:
            return

        data = {
            "devices": [
                {"pos": [d.position[0], d.position[1]]}
                for d in self.devices
            ],
            "blockages": [
                {"pos": [b.position[0], b.position[1]]}
                for b in self.blockages
            ],
            "settings": {
                "pwr":             self.cfg.pwr.value(),
                "noise":           self.cfg.noise.value(),
                "bw_sub6":         self.cfg.bw_sub6.value(),
                "bw_mm":           self.cfg.bw_mm.value(),
                "num_subchannel":  self.cfg.num_subchannel.value(),
                "num_beam":        self.cfg.num_beam.value(),
                "nframes":         self.cfg.nframes.value(),
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Layout + Settings", "", "JSON Files (*.json)"
        )
        if not path:
            return

        with open(path, "r") as f:
            data = json.load(f)

        # --- Clear existing items ---
        for item in list(self.devices):
            self.scene.removeItem(item)
            self.removeItem(item)
        for item in list(self.blockages):
            self.scene.removeItem(item)
            self.removeItem(item)
        self.devices.clear()
        self.blockages.clear()

        # --- Restore devices/blockages ---
        for d in data.get("devices", []):
            x, y = d["pos"]
            itm = DeviceItem(QPointF(x, y), self)
            self.devices.append(itm)
            self.scene.addItem(itm)
        for b in data.get("blockages", []):
            x, y = b["pos"]
            itm = BlockageItem(QPointF(x, y), self)
            self.blockages.append(itm)
            self.scene.addItem(itm)

        # --- Restore global config ---
        s = data.get("settings", {})
        if "pwr"            in s: self.cfg.pwr.setValue(s["pwr"])
        if "noise"          in s: self.cfg.noise.setValue(s["noise"])
        if "bw_sub6"        in s: self.cfg.bw_sub6.setValue(s["bw_sub6"])
        if "bw_mm"          in s: self.cfg.bw_mm.setValue(s["bw_mm"])
        if "num_subchannel" in s: self.cfg.num_subchannel.setValue(s["num_subchannel"])
        if "num_beam"       in s: self.cfg.num_beam.setValue(s["num_beam"])
        if "nframes"        in s: self.cfg.nframes.setValue(s["nframes"])
       

    def closeEvent(self, event):
        # Close all Matplotlib/QMainWindow graphs
        for gw in self.graph_windows.values():
            gw.close()
        if hasattr(self, 'choice_win_ref') and self.choice_win_ref:
            self.choice_win_ref.close()
        # Then proceed with normal shutdown
        super().closeEvent(event)



    def _scene_click(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.scenePos()
            items = self.scene.items(pos)
            for it in items:
                if isinstance(it, (DeviceItem, BlockageItem)):
                    QGraphicsScene.mouseDoubleClickEvent(self.scene, event)
                    return
            if self.mode == 'device': 
                itm = DeviceItem(pos, self)
                self.devices.append(itm)
            else: 
                itm = BlockageItem(pos, self)
                self.blockages.append(itm)
                
            self.scene.addItem(itm)
            return
        QGraphicsScene.mousePressEvent(self.scene, event)

    def _on_start(self):
        self.step_btn.setEnabled(True)
        self.auto_btn.setEnabled(True)
        self.start_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        
         # --- Packet Loss Rate Graph ---
        plr_gw = GraphWindow(title="Packet Loss Rate", xlabel="Frame", ylabel="PLR", fit_to_screen=False)
        for i in range(len(self.devices)):
            plr_gw.add_series(f"Device {i+1}", [0], [0])
        plr_gw.ax.axhline(y=0.1, color='dimgray', linestyle='--', linewidth=1.2, label='PLR Max')
        plr_gw.ax.legend() 
        plr_gw.canvas.draw()
        plr_gw.show()
        self.graph_windows['PLR'] = plr_gw
        
        
        # --- Choice Distribution Bar Graph ---
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_title("Choice Distribution")
        ax.set_xlabel("Device")
        ax.set_ylabel("Ratio")
        self.choice_window = {'canvas': canvas, 'ax': ax}
        self.choice_counts = [[0,0,0] for _ in self.devices]
        # layout in a new window
        choice_win = QMainWindow()
        choice_win.setWindowTitle("Choice Distribution")
        w = QWidget()
        choice_win.setCentralWidget(w)
        v = QVBoxLayout(w)
        v.addWidget(canvas)
        choice_win.show()
        self.choice_win_ref = choice_win
        
        self.setup_simulation()
        
    def _on_reset(self):
        # clear graphs
        for gw_name, gw in self.graph_windows.items(): 
            gw.clear()
            if gw_name == "PLR" and hasattr(gw, 'ax') and gw.ax:
                gw.ax.axhline(y=0.1, color='dimgray', linestyle='--', linewidth=1.2, label='PLR Max')
                gw.ax.legend() 
                gw.canvas.draw()

        self.choice_counts = [[0,0,0] for _ in self.devices]
        if self.choice_window:
            ax = self.choice_window['ax']; 
            ax.cla(); 
            ax.set_title("Choice Distribution"); 
            ax.set_xlabel("Device"); 
            ax.set_ylabel("Ratio"); 
            self.choice_window['canvas'].draw()

        self.setup_simulation()

        
    def setup_simulation(self):
        for d in self.devices: 
            d.history = {"sub6_success":[],"mmwave_success":[]}
        self.cur_frame = 0
        self.distances = [0] * len(self.devices)
        self.nblocks = [0] * len(self.devices)
        for i, device in enumerate(self.devices): # |device| * |blockage|
            sq = (device.position[0]**2 + device.position[1]**2)
            self.distances[i] = (sq ** 0.50)
            for blockage in self.blockages:
                cross = blockage.position[1] * device.position[0] - blockage.position[0] * device.position[1]
                if cross != 0:
                    continue
                dot = blockage.position[0] * device.position[0] + blockage.position[1] * device.position[1]
                if dot < 0:
                    continue
                if dot > sq:
                    continue
                self.nblocks[i] += 1
                
        self.plr = [0] * len(self.devices)
        self.RewList = []
        self.MetricList = []
        self.runnRew = 0
        if USE_RISK_AVERSE:
            self.qlearn = RiskAverseQLearning(len(self.devices), self.cfg.num_subchannel.value(), self.cfg.num_beam.value(), 4, FRAME_DURATION, PACKET_SIZE)
        else:
            if USE_DEEP_RISK:
                self.qlearn = RiskAverseDeepQLearning(len(self.devices), 
                    self.cfg.num_subchannel.value(), self.cfg.num_beam.value(), 4, 
                    FRAME_DURATION, PACKET_SIZE, 
                    replay_memory_capacity = 2000,
                    batch_size = 128,
                    dqn_hidden_layers=[20, 20, 20], 
                    St = 1000,
                    num_eta_levels=20,
                    eta_min_val=-len(self.devices)*2,
                    eta_max_val=len(self.devices))
            else:
                self.qlearn = DeepQLearning(len(self.devices), 
                    self.cfg.num_subchannel.value(), self.cfg.num_beam.value(), 4, 
                    FRAME_DURATION, PACKET_SIZE, 
                    replay_memory_capacity = 2000,
                    batch_size = 128,
                    dqn_hidden_layers=[20, 20, 20], 
                    St = 1000,
                )

    def _on_step(self):
        if self.cur_frame >= self.cfg.nframes.value(): 
            self.timer.stop()
            print("Simulation finished")
            
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for fig_name, graph_window_instance in self.graph_windows.items():
                if isinstance(graph_window_instance, GraphWindow): # Check if it's one of your custom graph windows
                    safe_fig_name = "".join(c if c.isalnum() else "_" for c in fig_name).lower()
                    filename = os.path.join(output_dir, f"{timestamp_str}_{safe_fig_name}_{SUFFIX_STR}.png")
                    try:
                        graph_window_instance.figure.savefig(filename)
                        print(f"Saved figure: {filename}")
                    except Exception as e:
                        print(f"Error saving figure {fig_name}: {e}")
            if hasattr(self, 'choice_window') and self.choice_window and 'canvas' in self.choice_window:
                # Assuming choice_window['canvas'].figure is the Matplotlib figure instance
                if hasattr(self.choice_window['canvas'], 'figure'):
                    safe_fig_name = "ChoiceDistribution".lower()
                    filename = os.path.join(output_dir, f"{timestamp_str}_{safe_fig_name}_{SUFFIX_STR}.png")
                    try:
                        self.choice_window['canvas'].figure.savefig(filename)
                        print(f"Saved figure: {filename}")
                    except Exception as e:
                        print(f"Error saving choice distribution figure: {e}")
                else:
                    print("Could not save choice distribution: Figure object not found in canvas.")
            else:
                print("Choice distribution window/figure not found for saving.")

            
            with open(os.path.join(output_dir, f'{timestamp_str}_log_{SUFFIX_STR}.txt'), 'w+') as f:
                f.write(f'Num of devices: {len(self.devices)}\n')
                f.write(f'PLR each devices: [ {" ".join(str(x) for x in self.plr)} ]\n')
                f.write(f'Avg Success: {(len(self.devices) - sum(self.plr)) / len(self.devices)}\n')
                f.write(f'Final Reward: {self.RewList[-1] / len(self.devices)}\n')
                f.write(f'Final Metric: {self.MetricList[-1]}\n')
                f.write(f'Data per Frame: \n\n')
                f.write('\n'.join([f'M: {m} | R: {r}' for m, r in zip(self.MetricList, self.RewList)]))
            self.RewList.clear()
            
            # print( self.qlearn.CC )
            return
        self.cur_frame += 1
        self.simulation()

    def _on_auto(self): 
        self.timer.start(0)
    
    def removeItem(self, item):
        assert item in self.devices or item in self.blockages, "Item should be in device or blockage list"
        if item in self.devices:
            self.devices.remove(item)
        else:
            self.blockages.remove(item)
    
    
    def _db_to_linear(self, db):
        return 10 ** (db / 10)
    def _dbm_to_linear(self, dbm):
        return self._db_to_linear(dbm-30)
    def _tx_beam_gain(self, theta, eps=0.1): #eq(4), no beta
        theta = abs(theta)
        return (2 * np.pi - (2 * np.pi - theta) * eps) / theta
    
    def simulation(self):
        num_devices = len(self.devices)
        achievable_rate = [(0, 0)] * num_devices
    
        # Calculate achievable rate for each devices
        P_tx_dbm = self.cfg.pwr.value()
        P_tx_W = self._dbm_to_linear(P_tx_dbm)
        
        noise_density_dbm_hz = self.cfg.noise.value()
        noise_density_W_hz = self._dbm_to_linear(noise_density_dbm_hz) # W/Hz

        W_sub_MHz = self.cfg.bw_sub6.value()
        W_sub_Hz = W_sub_MHz * 1e6
        N_subchannels = max(1, self.cfg.num_subchannel.value())
        W_sub_per_channel_Hz = W_sub_Hz / N_subchannels
        P_tx_sub_W = P_tx_W / N_subchannels
        
        W_mm_MHz = self.cfg.bw_mm.value()
        W_mm_Hz = W_mm_MHz * 1e6
        P_tx_mm_W = P_tx_W
        
        mmWave_tx_beamwidth_rad = 0.1
        mmWave_tx_sidelobe_gain = 0.1
        mmWave_rx_gain_lin = 1

        for i, dev in enumerate(self.devices):
            d = self.distances[i]
            
            # --- Sub-6GHz ---
            # Path Loss (Large-scale Fading)
            PL_sub_db = 38.5 + 30 * np.log10(d)
            PL_sub_lin = self._db_to_linear(-PL_sub_db)
            # Small-scale Fading (Rayleigh) - Power gain |h|^2
            h_small_scale_sub_power = np.random.rayleigh(scale=1.0)
            # Combined Channel Power Gain
            h_comb_sub_power = h_small_scale_sub_power * PL_sub_lin # |h_bkn(t)|^2 in Eq (1)
            # Noise Power in subchannel bandwidth
            noise_power_sub = noise_density_W_hz * W_sub_per_channel_Hz
            # SINR (I = 0)
            gamma_sub = (P_tx_sub_W * h_comb_sub_power) / noise_power_sub
            # Achievable Rate - Eq (5)
            rate_sub_bps = W_sub_per_channel_Hz * np.log2(1 + gamma_sub)

             # --- mmWave ---
            # Path Loss - based on blockage status
            is_blocked = self.nblocks[i] > 0
            if is_blocked: # NLoS
                shadowing_nlos_db = np.random.normal(0, 8.7)
                PL_mm_db = 72 + 29.2 * np.log10(d) + shadowing_nlos_db
            else: # LoS
                shadowing_los_db = np.random.normal(0, 5.8)
                PL_mm_db = 61.4 + 20 * np.log10(d) + shadowing_los_db
            PL_mm_lin = self._db_to_linear(-PL_mm_db)
            # Small-scale Fading (Rayleigh assumed for h_bkm^mW) - Power gain
            h_small_scale_mm_power = np.random.rayleigh(scale=1.0)
            # Tx Antenna Gain - Eq (4)
            G_tx_lin = self._tx_beam_gain(mmWave_tx_beamwidth_rad, mmWave_tx_sidelobe_gain) # G_b(theta, beta)
            # Rx Antenna Gain
            G_rx_lin = mmWave_rx_gain_lin # G_k^Rx
            # Combined Channel Power Gain - Eq (3)
            h_comb_mm_power = G_tx_lin * h_small_scale_mm_power * PL_mm_lin * G_rx_lin # h_bkm(theta, beta)
            # Noise Power in mmWave bandwidth
            noise_power_mm = noise_density_W_hz * W_mm_Hz
            # SINR (I = 0)
            gamma_mm = (P_tx_mm_W * h_comb_mm_power) / noise_power_mm
            # Achievable Rate - Eq (5)
            rate_mm_bps = W_mm_Hz * np.log2(1 + gamma_mm)

            achievable_rate[i] = (max(0, rate_sub_bps), max(0, rate_mm_bps))
            
        frame_duration = FRAME_DURATION 
        packet_size = PACKET_SIZE 
        achievable = [list(map(lambda x: int(x * frame_duration / packet_size), A)) for A in achievable_rate]
        #print("Achievable Rate: ", [list(map(float, A)) for A in achievable_rate])
        print("Achievable: ", achievable)
        
        if not USE_RISK_AVERSE and USE_DEEP_RISK:
            q_learn_act, eta_idx, safe = self.qlearn.get_current_action(self.cur_frame)
        else:
            q_learn_act, safe = self.qlearn.get_current_action(self.cur_frame)
        
        for d, a in enumerate(q_learn_act):
            self.choice_counts[d][a] += 1
        
        action = self.qlearn.map_action(q_learn_act)
        success = [[0, 0] for _ in range(num_devices)]
        if safe:
            success = [[min(act,achi) for act, achi in zip(a1, a2)] for a1, a2 in zip(action, achievable)]
        if not USE_RISK_AVERSE and USE_DEEP_RISK:
            reward = self.qlearn.update_to_new_state(success, self.cur_frame, q_learn_act, eta_idx, achievable_rate)
        else:
            reward = self.qlearn.update_to_new_state(success, self.cur_frame, q_learn_act, achievable_rate)
        
        metric = 0 
        for k in range(len(self.devices)):
            cur_psr = 1 if not sum(action[k]) else sum(success[k]) / sum(action[k])
            cur_plr = 1 - cur_psr
            old_plr = self.plr[k] * (self.cur_frame-1)
            self.plr[k] = (old_plr + cur_plr) / self.cur_frame
            metric += self.qlearn.PLR_req - self.plr[k]
        metric /= len(self.devices)
        self.RewList.append(reward)
        self.MetricList.append(metric)
        self.runnRew += reward
        
        print("Send: ", action)
        print("Recv: ", success)
        #print("Metric DeltaP: ", metric)
        
        if self.cur_frame % 100 == 0:
            
            gw=self.graph_windows.get("Metric"); gw.add_point("Metric", self.cur_frame, metric)
            gw=self.graph_windows.get("Reward"); gw.add_point("Reward", self.cur_frame, self.runnRew / max(1,self.cur_frame)  / max(1,len(self.devices)))
            
            plr_gw = self.graph_windows.get('PLR')
            for d, val in enumerate(self.plr):
                plr_gw.add_point(f"Device {d+1}", self.cur_frame, val)
        
            # update choice bar
            ax = self.choice_window['ax']
            ax.cla()
            num = len(self.devices)
            ind = np.arange(num)
            width = 0.2
            counts = np.array(self.choice_counts)
            for j in range(3):
                ax.bar(ind + (j-1)*width, counts[:, j], width, label=f"Choice {j}")
            ax.set_xticks(ind)
            ax.set_xticklabels([f"D{i+1}" for i in range(num)])
            ax.legend()
            self.choice_window['canvas'].draw()
            
            
        print(f"Stepped, frame: {self.cur_frame}")
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
