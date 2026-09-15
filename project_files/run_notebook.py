"""Execute the notebook with the project kernel and save all cell outputs."""

import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
# Keep Jupyter/IPython runtime files inside the project environment.
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".venv" / "jupyter-runtime"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".venv" / "ipython"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv" / "matplotlib"))

import nbformat
from nbclient import NotebookClient


def main():
    path = ROOT / "Walmart_Multi_Model_Time_Series_Forecasting.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)

    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            lines = [line for line in cell.source.splitlines()
                     if line.strip() and "=====" not in line]
            print(f"Cell {cell_index + 1}/{len(notebook.cells)}: {lines[0]}", flush=True)

    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="walmart-forecasting",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
        on_cell_start=progress,
    )
    started = time.monotonic()
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    count = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Saved notebook: {count} code cells completed in {time.monotonic() - started:.1f}s.")


if __name__ == "__main__":
    main()
