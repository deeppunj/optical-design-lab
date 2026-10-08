"""
ui/app.py  -  STAGE 6b: PySide6 lens designer
Run from the project root:   python ui/app.py
Adds: glass-catalog materials, wavelength selector, F/d/C chromatic
readout + colour spot diagram, save/load JSON, example lenses.
All physics lives in src/opticslab/designer.py (unit-tested).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

from opticslab import designer as dz
from opticslab.glass import LAMBDA_C, LAMBDA_D, LAMBDA_F
from opticslab.raytrace import at_image_plane, last_vertex_z, rms_spot_radius

HEADERS = ["Radius (mm)", "Thickness (mm)", "Glass after", "n at λ"]
LINES = {"d  (587.6 nm)": LAMBDA_D, "F  (486.1 nm)": LAMBDA_F,
         "C  (656.3 nm)": LAMBDA_C, "Custom": None}
COL = {"F": "tab:blue", "d": "tab:green", "C": "tab:red"}


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
        self.setWindowTitle("optical-design-lab - Stage 6b")
        self.resize(1300, 800)
        self._busy = False
        self._path = None

        # ---------------- prescription table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.itemChanged.connect(self.on_item_changed)
        b_add, b_del = QPushButton("Add surface"), QPushButton("Remove selected")
        b_add.clicked.connect(lambda: self.add_row(dz.SurfaceRow(0.0, 5.0, "AIR")) or self.update_all())
        b_del.clicked.connect(self.remove_row)
        btns = QHBoxLayout()
        btns.addWidget(b_add)
        btns.addWidget(b_del)

        # ---------------- beam controls
        self.epd = self._spin(0.1, 500, 10.0, 1, " mm")
        self.field = self._spin(-45, 45, 0.0, 2, " °")
        self.nrays = self._spin(3, 25, 11, 0)
        self.wl_combo = QComboBox()
        self.wl_combo.addItems(LINES.keys())
        self.wl = self._spin(300, 2500, LAMBDA_D, 4, " nm")
        self.chk_bf = QCheckBox("Move image plane to best focus")
        self.chk_bf.setChecked(True)
        self.chk_col = QCheckBox("Spot diagram: show F / d / C colours")
        form = QFormLayout()
        form.addRow("Entrance pupil Ø", self.epd)
        form.addRow("Field angle", self.field)
        form.addRow("Rays in layout", self.nrays)
        form.addRow("Wavelength line", self.wl_combo)
        form.addRow("Wavelength", self.wl)
        form.addRow(self.chk_bf)
        form.addRow(self.chk_col)
        box = QGroupBox("Beam")
        box.setLayout(form)
        for w in (self.epd, self.field, self.nrays):
            w.valueChanged.connect(self.update_all)
        self.chk_bf.stateChanged.connect(self.update_all)
        self.chk_col.stateChanged.connect(self.update_all)
        self.wl_combo.currentTextChanged.connect(self.on_line)
        self.wl.valueChanged.connect(self.on_wl)

        self.readout = QLabel()
        self.readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.readout.setStyleSheet("font-family: Menlo, monospace;")

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("<b>Prescription</b> (radius 0 = flat; last "
                            "thickness = distance to image)"))
        ll.addWidget(self.table, 1)
        ll.addLayout(btns)
        ll.addWidget(box)
        ll.addWidget(self.readout)

        # ---------------- plots
        self.p_layout, self.p_spot, self.p_fan = Plot(), Plot(), Plot()
        tabs = QTabWidget()
        tabs.addTab(self.p_layout, "Layout")
        tabs.addTab(self.p_spot, "Spot diagram")
        tabs.addTab(self.p_fan, "Ray fan")
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([500, 800])
        self.setCentralWidget(split)
        self.statusBar().showMessage("Ready")
        self._build_menu()
        self.set_rows(dz.EXAMPLES["Singlet N-BK7 (50/-50)"])

    # ------------------------------------------------------------- menu
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        for text, key, fn in (("&Open…", QKeySequence.Open, self.open_file),
                              ("&Save", QKeySequence.Save, self.save_file),
                              ("Save &As…", QKeySequence.SaveAs, self.save_as)):
            a = QAction(text, self)
            a.setShortcut(key)
            a.triggered.connect(fn)
            m.addAction(a)
        ex = self.menuBar().addMenu("&Examples")
        for name in dz.EXAMPLES:
            a = QAction(name, self)
            a.triggered.connect(lambda _=False, n=name: self.load_example(n))
            ex.addAction(a)

    def load_example(self, name):
        self._path = None
        self.set_rows(dz.EXAMPLES[name])

    def open_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open design", "", "Design (*.json)")
        if p:
            self.open_path(p)

    def open_path(self, p):
        try:
            rows, epd, fld, wl = dz.load_design(p)
        except Exception as e:
            QMessageBox.warning(self, "Open failed", str(e))
            return
        self._path = p
        self._busy = True
        self.epd.setValue(epd)
        self.field.setValue(fld)
        self.wl.setValue(wl)
        self._sync_combo()
        self._busy = False
        self.set_rows(rows)

    def save_file(self):
        if self._path is None:
            return self.save_as()
        self.save_path(self._path)

    def save_as(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save design", "design.json", "Design (*.json)")
        if p:
            if not p.endswith(".json"):
                p += ".json"
            self.save_path(p)

    def save_path(self, p):
        try:
            dz.save_design(p, self.read_rows(), self.epd.value(),
                           self.field.value(), self.wl.value())
            self._path = p
            self.statusBar().showMessage(f"Saved {p}")
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))

    # -------------------------------------------------------- wavelength
    def on_line(self, text):
        wl = LINES.get(text)
        if wl is not None:
            self.wl.blockSignals(True)
            self.wl.setValue(wl)
            self.wl.blockSignals(False)
        self.update_all()

    def on_wl(self, _):
        self._sync_combo()
        self.update_all()

    def _sync_combo(self):
        match = next((k for k, v in LINES.items()
                      if v is not None and abs(v - self.wl.value()) < 0.05), "Custom")
        self.wl_combo.blockSignals(True)
        self.wl_combo.setCurrentText(match)
        self.wl_combo.blockSignals(False)

    # ------------------------------------------------------------- table
    @staticmethod
    def _spin(lo, hi, val, dec, suffix=""):
        s = QSpinBox() if dec == 0 else QDoubleSpinBox()
        if dec:
            s.setDecimals(dec)
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSuffix(suffix)
        return s

    def set_rows(self, rows):
        self._busy = True
        self.table.setRowCount(0)
        for r in rows:
            self.add_row(r)
        self._busy = False
        self.refresh_n()
        self.update_all()

    def add_row(self, row):
        was = self._busy
        self._busy = True
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(f"{row.radius:g}"))
        self.table.setItem(r, 1, QTableWidgetItem(f"{row.thickness:g}"))
        combo = QComboBox()
        combo.addItems(dz.GLASS_CHOICES)
        combo.setCurrentText(row.glass)
        combo.currentTextChanged.connect(lambda _t, c=combo: self.on_glass(c))
        self.table.setCellWidget(r, 2, combo)
        self.table.setItem(r, 3, QTableWidgetItem(f"{row.n_custom:.5f}"))
        self._busy = was
        if not was:
            self.refresh_n()

    def remove_row(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.table.rowCount():
            rows = [self.table.rowCount() - 1]
        for r in rows:
            self.table.removeRow(r)
        self.update_all()

    def on_glass(self, combo):
        if self._busy:
            return
        self.refresh_n(keep_custom_from=combo)
        self.update_all()

    def on_item_changed(self, _item):
        if not self._busy:
            self.update_all()

    def refresh_n(self, keep_custom_from=None):
        """Show n(λ) for catalog glasses (read-only); Custom stays editable."""
        was, self._busy = self._busy, True
        wl = self.wl.value()
        for r in range(self.table.rowCount()):
            combo = self.table.cellWidget(r, 2)
            item = self.table.item(r, 3)
            if combo.currentText() == dz.CUSTOM:
                item.setFlags(item.flags() | Qt.ItemIsEditable)
            else:
                item.setText(f"{dz.CATALOG[combo.currentText()].n(wl):.5f}")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self._busy = was

    @staticmethod
    def _num(text, what, r):
        try:
            return float(text.strip())
        except ValueError:
            raise ValueError(f"Row {r + 1}: invalid {what} '{text}'")

    def read_rows(self):
        rows = []
        for r in range(self.table.rowCount()):
            g = self.table.cellWidget(r, 2).currentText()
            rows.append(dz.SurfaceRow(
                self._num(self.table.item(r, 0).text(), "radius", r),
                self._num(self.table.item(r, 1).text(), "thickness", r),
                g,
                self._num(self.table.item(r, 3).text(), "index", r) if g == dz.CUSTOM else 1.0))
        dz.validate(rows)
        return rows

    # ------------------------------------------------------------ update
    def update_all(self, *_):
        if self._busy:
            return
        try:
            self.refresh_n()
            rows = self.read_rows()
            wl, epd, th = self.wl.value(), self.epd.value(), self.field.value()
            px, py = dz.hex_pupil(epd, 6)
            surfs, o, pts, dd = dz.trace_bundle(rows, wl, px, py, th)
            efl, bfl = dz.paraxial(rows, wl)
            z_last = last_vertex_z(surfs)
            z_img = dz.image_plane_z(rows)
            if self.chk_bf.isChecked():
                lo, hi = dz.focus_search_range(rows, bfl)
                z_img = dz.best_focus_z(pts, dd, lo, hi)
            img = at_image_plane(pts, dd, z_img)
            rms = rms_spot_radius(img[:, :2])

            self.draw_layout(rows, wl, epd, th, z_img)
            self.draw_spot(rows, img, rms, px, py, th, z_img)
            self.draw_fan(rows, wl, epd, th, z_img)

            chrom = dz.chromatic_table(rows)
            col = chrom[0]["bfl"] - chrom[2]["bfl"]
            self.readout.setText(
                f"λ              : {wl:9.2f} nm\n"
                f"EFL (paraxial) : {efl:9.3f} mm\n"
                f"BFL (paraxial) : {bfl:9.3f} mm\n"
                f"Image plane z  : {z_img:9.3f} mm\n"
                f"RMS spot radius: {rms * 1000:9.2f} µm\n"
                f"f/#            : {abs(efl) / epd:9.2f}\n"
                f"-- chromatic (paraxial) --\n"
                f"EFL F/d/C      : {chrom[0]['efl']:.3f} / {chrom[1]['efl']:.3f} / {chrom[2]['efl']:.3f}\n"
                f"Axial colour   : {col:9.3f} mm  (BFL_F − BFL_C)")
            self.statusBar().showMessage("Updated")
        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")

    # ------------------------------------------------------------- plots
    def draw_layout(self, rows, wl, epd, th, z_img):
        ax = self.p_layout.ax
        ax.clear()
        n = int(self.nrays.value())
        py = np.linspace(-epd / 2, epd / 2, n)
        surfs, o, pts, dd = dz.trace_bundle(rows, wl, np.zeros(n), py, th)
        img = at_image_plane(pts, dd, z_img)
        for k in range(n):
            path = np.array([o[k]] + [pts[i, k] for i in range(len(surfs))] + [img[k]])
            ax.plot(path[:, 2], path[:, 1], lw=0.9, color="tab:red", alpha=0.8)
        semi = epd / 2 * 1.1
        h = np.linspace(-semi, semi, 120)
        zv = 0.0
        for i, r in enumerate(rows):
            x1 = zv + dz.sag_profile(r.radius, h)
            ax.plot(x1, h, "k", lw=1.3)
            if i + 1 < len(rows) and r.index(wl) > 1.0001:
                x2 = zv + r.thickness + dz.sag_profile(rows[i + 1].radius, h)
                ax.fill_betweenx(h, x1, x2, color="#9ec5e8", alpha=0.45, lw=0)
                ax.plot([x1[0], x2[0]], [h[0], h[0]], "k", lw=1.3)
                ax.plot([x1[-1], x2[-1]], [h[-1], h[-1]], "k", lw=1.3)
            zv += r.thickness
        ax.axvline(z_img, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(f"Layout - λ {wl:.1f} nm, field {th:g}° (dashed = image plane)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        self.p_layout.canvas.draw_idle()

    def draw_spot(self, rows, img, rms, px, py, th, z_img):
        ax = self.p_spot.ax
        ax.clear()
        ref = img[:, :2]
        ok = ~np.isnan(ref).any(axis=1)
        c = ref[ok].mean(axis=0) if ok.any() else np.zeros(2)
        if self.chk_col.isChecked():
            for name, w in (("F", LAMBDA_F), ("d", LAMBDA_D), ("C", LAMBDA_C)):
                _, _, p2, d2 = dz.trace_bundle(rows, w, px, py, th)
                xy = at_image_plane(p2, d2, z_img)[:, :2]
                ax.scatter((xy[:, 0] - c[0]) * 1000, (xy[:, 1] - c[1]) * 1000,
                           s=9, color=COL[name], label=name, alpha=0.7)
            ax.legend(loc="upper right")
            ttl = f"Spot diagram F/d/C at z = {z_img:.2f} mm (d RMS {rms * 1000:.2f} µm)"
        else:
            ax.scatter((ref[:, 0] - c[0]) * 1000, (ref[:, 1] - c[1]) * 1000, s=10)
            ttl = f"Spot diagram - RMS radius {rms * 1000:.2f} µm"
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_title(ttl)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        self.p_spot.canvas.draw_idle()

    def draw_fan(self, rows, wl, epd, th, z_img):
        ax = self.p_fan.ax
        ax.clear()
        n = 61
        py = np.linspace(-epd / 2, epd / 2, n)
        for name, w in (("F", LAMBDA_F), ("d", LAMBDA_D), ("C", LAMBDA_C)):
            _, _, pts, dd = dz.trace_bundle(rows, w, np.zeros(n), py, th)
            yi = at_image_plane(pts, dd, z_img)[:, 1]
            ax.plot(py, (yi - yi[n // 2]) * 1000, lw=1.4, color=COL[name], label=name)
        ax.axhline(0, color="gray", lw=0.6)
        ax.legend(loc="upper left")
        ax.set_xlabel("Pupil height (mm)")
        ax.set_ylabel("Transverse error (µm)")
        ax.set_title("Ray fan (meridional) at F / d / C")
        ax.grid(alpha=0.3)
        self.p_fan.canvas.draw_idle()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
