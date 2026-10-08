"""
ui/app.py  -  STAGE 6: PySide6 front-end for optical-design-lab
Run from the project root:   python ui/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QMainWindow, QPushButton, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

from opticslab.raytrace import (
    TraceSurface, trace, at_image_plane, last_vertex_z, rms_spot_radius,
)

# ----------------------------------------------------------------- materials
MATERIALS = {
    "Air": 1.0,
    "N-BK7": 1.5168,
    "Fused silica": 1.4585,
    "N-SF11": 1.7847,
    "N-SF5": 1.6727,
    "F2": 1.6200,
    "Custom": None,
}
HEADERS = ["Radius (mm)", "Thickness (mm)", "Material", "n after"]
Z_START = -10.0   # rays start 10 mm before the first vertex


# ------------------------------------------------------------- engine helpers
def build_surfaces(rows):
    """rows: list of (radius, thickness, n_after) -> list[TraceSurface].
    radius 0 or inf = flat. ADAPT HERE if TraceSurface has other fields."""
    out = []
    for R, t, n in rows:
        R = np.inf if (R == 0 or not np.isfinite(R)) else R
        out.append(TraceSurface(radius=R, thickness=t, n_after=n))
    return out


def make_rays(py, px, theta_deg):
    """Rays that hit the first vertex plane (z=0) at pupil (px, py),
    travelling at field angle theta in the y-z plane."""
    th = np.radians(theta_deg)
    d0 = np.array([0.0, np.sin(th), np.cos(th)])
    n = len(py)
    dirs = np.tile(d0, (n, 1))
    o = np.zeros((n, 3))
    o[:, 0] = px
    o[:, 1] = py + Z_START * np.tan(th)       # y at z=0 equals py
    o[:, 2] = Z_START
    return o, dirs


def paraxial(rows, n_start=1.0):
    """Paraxial EFL and BFL (mm) from a parallel ray, y1 = 1."""
    y, u, n_prev = 1.0, 0.0, n_start
    for i, (R, t, n) in enumerate(rows):
        c = 0.0 if (R == 0 or not np.isfinite(R)) else 1.0 / R
        u = (n_prev * u - y * (n - n_prev) * c) / n
        n_prev = n
        if i < len(rows) - 1:
            y += t * u
    if abs(u) < 1e-12:
        return np.inf, np.inf
    return -1.0 / u, -y / u


def spot_pupil(epd, rings=6):
    """Hexapolar-style pupil sampling inside a circle of diameter epd."""
    px, py = [0.0], [0.0]
    for r in range(1, rings + 1):
        k = 6 * r
        a = np.linspace(0, 2 * np.pi, k, endpoint=False)
        px += list(r / rings * epd / 2 * np.cos(a))
        py += list(r / rings * epd / 2 * np.sin(a))
    return np.array(px), np.array(py)


def best_focus_z(pts, dirs, z_lo, z_hi):
    """Scan the image distance for the minimum RMS spot."""
    zs = np.linspace(z_lo, z_hi, 600)
    vals = [rms_spot_radius(at_image_plane(pts, dirs, z)[:, :2]) for z in zs]
    vals = np.array(vals)
    if np.all(np.isnan(vals)):
        return z_lo
    i = int(np.nanargmin(vals))
    zs2 = np.linspace(zs[max(i - 1, 0)], zs[min(i + 1, len(zs) - 1)], 200)
    v2 = [rms_spot_radius(at_image_plane(pts, dirs, z)[:, :2]) for z in zs2]
    return float(zs2[int(np.nanargmin(v2))])


def sag_profile(R, h):
    if R == 0 or not np.isfinite(R):
        return np.zeros_like(h)
    return R - np.sign(R) * np.sqrt(np.maximum(R * R - h * h, 0.0))


# ------------------------------------------------------------------- widgets
class Plot(QWidget):
    def __init__(self):
        super().__init__()
        self.fig = Figure(figsize=(6, 4), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = Canvas(self.fig)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("optical-design-lab  -  Stage 6")
        self.resize(1250, 780)
        self._busy = False

        # --- left panel: prescription + controls
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.itemChanged.connect(self.on_item_changed)

        btn_add = QPushButton("Add surface")
        btn_del = QPushButton("Remove selected")
        btn_ex = QPushButton("Load example")
        btn_add.clicked.connect(lambda: self.add_row(0.0, 5.0, "Air"))
        btn_del.clicked.connect(self.remove_row)
        btn_ex.clicked.connect(self.load_example)
        btns = QHBoxLayout()
        for b in (btn_add, btn_del, btn_ex):
            btns.addWidget(b)

        self.epd = self._spin(0.1, 500, 10.0, 1, " mm")
        self.field = self._spin(-45, 45, 0.0, 2, " deg")
        self.nrays = self._spin(3, 25, 11, 0)
        self.chk_bf = QCheckBox("Move image plane to best focus")
        self.chk_bf.setChecked(True)
        form = QFormLayout()
        form.addRow("Entrance pupil dia.", self.epd)
        form.addRow("Field angle", self.field)
        form.addRow("Rays in fan", self.nrays)
        form.addRow(self.chk_bf)
        box = QGroupBox("Beam")
        box.setLayout(form)
        for w in (self.epd, self.field, self.nrays):
            w.valueChanged.connect(self.update_all)
        self.chk_bf.stateChanged.connect(self.update_all)

        self.readout = QLabel()
        self.readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.readout.setStyleSheet("font-family: Menlo, monospace;")

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("<b>Prescription</b>  (radius 0 = flat; "
                            "last thickness = distance to image)"))
        ll.addWidget(self.table, 1)
        ll.addLayout(btns)
        ll.addWidget(box)
        ll.addWidget(self.readout)

        # --- right panel: plots
        self.p_layout, self.p_spot, self.p_fan = Plot(), Plot(), Plot()
        tabs = QTabWidget()
        tabs.addTab(self.p_layout, "Layout")
        tabs.addTab(self.p_spot, "Spot diagram")
        tabs.addTab(self.p_fan, "Ray fan")

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([480, 770])
        self.setCentralWidget(split)
        self.statusBar().showMessage("Ready")

        self.load_example()

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _spin(lo, hi, val, dec, suffix=""):
        s = QSpinBox() if dec == 0 else QDoubleSpinBox()
        if dec:
            s.setDecimals(dec)
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSuffix(suffix)
        return s

    def add_row(self, R, t, material, n=None):
        self._busy = True
        r = self.table.rowCount()
        self.table.insertRow(r)
        n = MATERIALS.get(material) if n is None else n
        n = 1.0 if n is None else n
        self.table.setItem(r, 0, QTableWidgetItem(f"{R:g}"))
        self.table.setItem(r, 1, QTableWidgetItem(f"{t:g}"))
        combo = QComboBox()
        combo.addItems(MATERIALS.keys())
        combo.setCurrentText(material)
        combo.currentTextChanged.connect(lambda _txt, c=combo: self.on_material(c))
        self.table.setCellWidget(r, 2, combo)
        self.table.setItem(r, 3, QTableWidgetItem(f"{n:.5f}"))
        self._busy = False
        self.update_all()

    def remove_row(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.table.rowCount():
            rows = [self.table.rowCount() - 1]
        for r in rows:
            self.table.removeRow(r)
        self.update_all()

    def row_of(self, combo):
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 2) is combo:
                return r
        return -1

    def on_material(self, combo):
        r = self.row_of(combo)
        n = MATERIALS[combo.currentText()]
        if r >= 0 and n is not None:
            self._busy = True
            self.table.item(r, 3).setText(f"{n:.5f}")
            self._busy = False
        self.update_all()

    def on_item_changed(self, item):
        if self._busy:
            return
        if item.column() == 3:                       # manual n -> "Custom"
            combo = self.table.cellWidget(item.row(), 2)
            if combo is not None and combo.currentText() != "Custom":
                nval = self._num(item.text(), None)
                if nval is None or MATERIALS[combo.currentText()] != nval:
                    combo.blockSignals(True)
                    combo.setCurrentText("Custom")
                    combo.blockSignals(False)
        self.update_all()

    def load_example(self):
        self._busy = True
        self.table.setRowCount(0)
        self._busy = False
        self.add_row(50.0, 5.0, "N-BK7")
        self.add_row(-50.0, 45.0, "Air")

    @staticmethod
    def _num(text, default):
        try:
            return float(text.strip().lower().replace("inf", "inf"))
        except ValueError:
            return default

    def read_rows(self):
        rows = []
        for r in range(self.table.rowCount()):
            R = self._num(self.table.item(r, 0).text(), None)
            t = self._num(self.table.item(r, 1).text(), None)
            n = self._num(self.table.item(r, 3).text(), None)
            if R is None or t is None or n is None or n <= 0:
                raise ValueError(f"Row {r + 1}: invalid number")
            rows.append((R, t, n))
        if not rows:
            raise ValueError("Add at least one surface")
        return rows

    # --------------------------------------------------------------- update
    def update_all(self, *_):
        if self._busy:
            return
        try:
            rows = self.read_rows()
            surfs = build_surfaces(rows)
            epd, th = self.epd.value(), self.field.value()
            z_last = last_vertex_z(surfs)

            # spot-diagram bundle
            px, py = spot_pupil(epd)
            o, d = make_rays(py, px, th)
            pts, dd = trace(o, d, surfs)

            z_img = z_last + rows[-1][1]
            if self.chk_bf.isChecked():
                z_img = best_focus_z(pts, dd, z_last + 0.2, z_last + 6 * max(abs(z_img - z_last), 50))
            efl, bfl = paraxial(rows)
            img = at_image_plane(pts, dd, z_img)
            rms = rms_spot_radius(img[:, :2])

            self.draw_layout(rows, surfs, epd, th, z_img)
            self.draw_spot(img, rms)
            self.draw_fan(surfs, epd, th, z_img)

            self.readout.setText(
                f"EFL (paraxial)  : {efl:10.3f} mm\n"
                f"BFL (paraxial)  : {bfl:10.3f} mm\n"
                f"Image plane z   : {z_img:10.3f} mm\n"
                f"RMS spot radius : {rms * 1000:10.2f} um\n"
                f"f/#             : {abs(efl) / epd:10.2f}"
            )
            self.statusBar().showMessage("Updated")
        except Exception as e:                       # keep GUI alive
            self.statusBar().showMessage(f"Error: {e}")

    # ---------------------------------------------------------------- plots
    def draw_layout(self, rows, surfs, epd, th, z_img):
        ax = self.p_layout.ax
        ax.clear()
        n = int(self.nrays.value())
        py = np.linspace(-epd / 2, epd / 2, n)
        o, d = make_rays(py, np.zeros(n), th)
        pts, dd = trace(o, d, surfs)
        img = at_image_plane(pts, dd, z_img)

        for k in range(n):
            path = [o[k]] + [pts[i, k] for i in range(len(surfs))] + [img[k]]
            path = np.array(path)
            ax.plot(path[:, 2], path[:, 1], lw=0.9)

        zv = 0.0
        h = np.linspace(-epd / 2 * 1.1, epd / 2 * 1.1, 120)
        for (R, t, _), i in zip(rows, range(len(rows))):
            ax.plot(zv + sag_profile(R, h), h, "k", lw=1.4)
            zv += t
        # draw glass edges between consecutive surfaces if n > 1.0001
        zv = 0.0
        for i in range(len(rows) - 1):
            if rows[i][2] > 1.0001:
                R1, t1 = rows[i][0], rows[i][1]
                R2 = rows[i + 1][0]
                for hh in (-epd / 2 * 1.1, epd / 2 * 1.1):
                    ax.plot([zv + sag_profile(R1, np.array([hh]))[0],
                             zv + t1 + sag_profile(R2, np.array([hh]))[0]],
                            [hh, hh], "k", lw=1.4)
            zv += rows[i][1]

        ax.axvline(z_img, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(f"Layout  (field {th:g} deg, dashed = image plane)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        self.p_layout.canvas.draw_idle()

    def draw_spot(self, img, rms):
        ax = self.p_spot.ax
        ax.clear()
        xy = img[:, :2]
        ok = ~np.isnan(xy).any(axis=1)
        c = xy[ok].mean(axis=0) if ok.any() else (0, 0)
        ax.scatter((xy[:, 0] - c[0]) * 1000, (xy[:, 1] - c[1]) * 1000, s=10)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_title(f"Spot diagram - RMS radius {rms * 1000:.2f} um")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        self.p_spot.canvas.draw_idle()

    def draw_fan(self, surfs, epd, th, z_img):
        ax = self.p_fan.ax
        ax.clear()
        n = 61
        py = np.linspace(-epd / 2, epd / 2, n)
        o, d = make_rays(py, np.zeros(n), th)
        pts, dd = trace(o, d, surfs)
        y_img = at_image_plane(pts, dd, z_img)[:, 1]
        chief = y_img[n // 2]
        ax.plot(py, (y_img - chief) * 1000, lw=1.5)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xlabel("Pupil height (mm)")
        ax.set_ylabel("Transverse error (um)")
        ax.set_title("Ray fan (meridional)")
        ax.grid(alpha=0.3)
        self.p_fan.canvas.draw_idle()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
