"""Rebuild and verify the paper outputs. Run from any working directory."""
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "outputs"
out.mkdir(exist_ok=True)
env = dict(os.environ, MPLBACKEND="Agg", OPENBLAS_NUM_THREADS="1")
for script in ["analysis/rebuild_detectgpt.py", "analysis/rebuild_gptzero.py", "analysis/check_text_fluoroscopy.py",
               "analysis/reproduce.py", "analysis/performance_tables.py",
               "experiments/rq6/evaluate.py", "figures/render.py", "analysis/verify.py"]:
    subprocess.run([sys.executable, str(root/script)], cwd=root, env=env, check=True)
