"""Headless smoke test of the Stage 6b GUI (skipped if PySide6 is missing)."""
import importlib.util
import os
from pathlib import Path
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("matplotlib")
try:
    from PySide6.QtWidgets import QApplication
except ImportError as exc:          # e.g. PySide6 missing or no libGL
    pytest.skip(f"Qt unavailable: {exc}", allow_module_level=True)

APP = Path(__file__).resolve().parents[1] / "ui" / "app.py"

@pytest.fixture(scope="module")
def win():
    qapp = QApplication.instance() or QApplication([])
    spec = importlib.util.spec_from_file_location("lens_app", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    w = mod.MainWindow()
    yield w, mod
    w.close()

def test_starts_and_reports_efl(win):
    w, _ = win
    assert "EFL" in w.readout.text() and "Error" not in w.statusBar().currentMessage()

def test_every_example_updates_without_error(win):
    w, mod = win
    for name in mod.dz.EXAMPLES:
        w.load_example(name)
        assert w.statusBar().currentMessage() == "Updated", name

def test_wavelength_and_colour_mode(win):
    w, mod = win
    w.load_example("Achromat N-BK7/N-F2")
    w.chk_col.setChecked(True)
    w.wl_combo.setCurrentText("F  (486.1 nm)")
    assert abs(w.wl.value() - 486.1327) < 1e-3
    w.field.setValue(3.0)
    assert w.statusBar().currentMessage() == "Updated"

def test_save_and_open_roundtrip(win, tmp_path):
    w, mod = win
    w.load_example("Achromat N-BK7/N-F2")
    p = str(tmp_path / "d.json")
    w.save_path(p)
    w.load_example("Singlet N-BK7 (50/-50)")
    w.open_path(p)
    assert w.table.rowCount() == 3
    assert w.statusBar().currentMessage() == "Updated"

def test_bad_input_does_not_crash(win):
    w, _ = win
    w.table.item(0, 0).setText("abc")
    assert w.statusBar().currentMessage().startswith("Error")
    w.table.item(0, 0).setText("50")
    assert w.statusBar().currentMessage() == "Updated"
