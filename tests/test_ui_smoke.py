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

def test_optimizer_controls_and_variable_flags(win):
    """Stage 7 starts with controls disabled appropriately and reads flags."""
    w, mod = win
    w.load_example("Achromat N-BK7/N-F2")
    assert w.table.columnCount() == 6
    assert not w.b_stop.isEnabled()
    assert not w.b_undo.isEnabled()
    w.table.item(0, 4).setCheckState(mod.Qt.Checked)
    w.table.item(2, 4).setCheckState(mod.Qt.Checked)
    w.table.item(2, 5).setCheckState(mod.Qt.Checked)
    assert w.read_flags() == ([0, 2], [2])


def test_optimizer_runs_and_undo_restores_design(win):
    """Run a deliberately small Stage 7 optimisation without a modal dialog.

    We perturb the image distance, select it as the only variable, and give an
    EFL target.  The test checks that the merit tab/report appear and that Undo
    returns precisely to the prescription before the optimisation.
    """
    w, mod = win
    w.load_example("Achromat N-BK7/N-F2")
    original = w.read_rows()
    w.table.item(2, 1).setText("85")             # intentionally wrong image plane
    before = w.read_rows()
    w.set_flags([], [2])
    w.o_efl_on.setChecked(True)
    w.o_efl.setValue(100.0)
    w.o_iter.setValue(30)
    w.run_opt()
    assert "merit" in w.o_report.text()
    assert w.tabs.currentWidget() is w.p_merit
    assert w.history.can_undo()
    changed = w.read_rows()
    assert changed != before
    w.undo_opt()
    assert w.read_rows() == before
    w.redo_opt()
    assert w.read_rows() == changed
