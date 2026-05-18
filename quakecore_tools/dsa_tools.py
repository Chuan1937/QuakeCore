"""
DSA (Depth-Scanning Algorithm) integration for QuakeCore.

Determines focal depth of local/regional earthquakes using 3-component waveforms.
Runs the original DSA_1.0.3.py script and collects generated plots for frontend display.

Reference:
    Yuan, J., Kao, H., & Yu, J. (2020). Depth-Scanning Algorithm. JGR Solid Earth, 125(7).
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


EXAMPLE_MAP = {
    "Example1": "Example1_PAPER-Section3.2",
    "Example2": "Example2_2014-10-07-mw40-Oklahoma",
    "Example3": "Example3_2014-10-10-mw43_Oklahoma",
    "Example4": "Example4_2014-02-15-mw41-Carolina",
    "Example5": "Example5_2014-02-16-mw30-Carolina",
    "Example6": "Example6_2014-03-29-mw41-California",
    "Example7": "Example7_2018-08-29-mw44-California",
    "Example8": "Example8_2014-10-07-16-51-13",
}

_PLOT_PATCH_PREAMBLE = r'''
# ------------------------------------------------------------
# QuakeCore injection: save all plots to file for frontend display
import os as _qc_os, matplotlib as _qc_mpl
_qc_mpl.use("Agg")
import matplotlib.pyplot as _qc_plt
_QC_PLOT_DIR = _qc_os.environ.get("DSA_PLOT_DIR", _qc_os.path.join(_qc_os.path.dirname(_qc_os.path.abspath(__file__)), "plots"))
_qc_os.makedirs(_QC_PLOT_DIR, exist_ok=True)
_QC_PLOT_COUNTER = [0]
_QC_SAVE_DPI = 150

def _qc_save_figure(fig=None, prefix="dsa"):
    _QC_PLOT_COUNTER[0] += 1
    fname = _qc_os.path.join(_QC_PLOT_DIR, f"{prefix}_{_QC_PLOT_COUNTER[0]:04d}.png")
    (fig or _qc_plt.gcf()).savefig(fname, dpi=_QC_SAVE_DPI, bbox_inches="tight")
    print(f"[QuakeCore] Saved figure: {fname}")

# Patch plt.show() and pltDebug.show() to save figures
import matplotlib.pyplot
_orig_plt_show = matplotlib.pyplot.show
matplotlib.pyplot.show = lambda *a, **kw: _qc_save_figure(prefix="dsa_main")
try:
    import matplotlib.pyplot as pltDebug
    _orig_pltDebug_show = pltDebug.show
    pltDebug.show = lambda *a, **kw: _qc_save_figure(prefix="dsa_debug")
except: pass
# ------------------------------------------------------------
'''


def _get_data_dirs():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "professional" / "dsa"
    examples_dir = data_dir / "examples"
    dsa_script = data_dir / "DSA_1.0.3.py"
    return repo_root, data_dir, examples_dir, dsa_script


def _collect_plots(results_dir: str) -> List[str]:
    """Collect all plot files from the results directory."""
    plots = []
    results_path = Path(results_dir)
    if not results_path.exists():
        return plots
    for ext in ('*.png', '*.svg', '*.jpg'):
        for f in sorted(results_path.rglob(ext)):
            plots.append(str(f))
    return plots


def run_dsa_example(
    example_name: str = "Example1",
    data_dir: Optional[str] = None,
    verbose: bool = False,
    timeout: int = 1800,
) -> Dict:
    """Run DSA on a specified example and return results with plot paths.

    Args:
        example_name: "Example1" - "Example8"
        data_dir: Override base data directory
        verbose: Print detailed output
        timeout: Subprocess timeout in seconds

    Returns:
        Dict with status, focal_depth_km, num_stations, plots[], results_file, stdout
    """
    import pandas as pd
    repo_root, default_data_dir, examples_dir, dsa_script = _get_data_dirs()

    if data_dir:
        examples_dir = Path(data_dir)

    example_subdir = EXAMPLE_MAP.get(example_name, example_name)
    example_path = examples_dir / example_subdir

    if not example_path.exists():
        return {"error": f"Example directory not found: {example_path}"}
    if not dsa_script.exists():
        return {"error": f"DSA script not found: {dsa_script}"}

    settings_files = sorted(example_path.parent.glob(f"DSA_SETTINGS_{example_name}*.txt"))
    if not settings_files:
        settings_files = sorted(example_path.glob("DSA_SETTINGS*.txt"))
    if not settings_files:
        return {"error": f"No settings file for {example_name}"}

    settings = pd.read_csv(str(settings_files[0]), sep=r'\s+', index_col='PARAMETER')
    vel_model = str(settings.VALUE.loc['velModel'])
    resolved_data_path = str(example_path) + "/"

    # Verify data
    if not list(example_path.glob("*.sac")) and not list(example_path.glob("*.mseed")):
        return {"error": f"No waveform files in {resolved_data_path}"}
    if not (example_path / f"{vel_model}.nd").exists():
        return {"error": f"Velocity model not found: {vel_model}.nd"}

    # Remove old results (keep fresh)
    old_results = example_path / "results"
    if old_results.exists():
        shutil.rmtree(str(old_results))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepare settings file
        settings_content = f"""PARAMETER\t\tVALUE
dataPath\t\t\t{resolved_data_path}
velModel\t\t\t{vel_model}
arrTimeDiffTole \t\t{settings.VALUE.loc['arrTimeDiffTole']}
ccThreshold\t\t{settings.VALUE.loc['ccThreshold']}
frequencyFrom\t\t{settings.VALUE.loc['frequencyFrom']}
frequencyTo\t\t{settings.VALUE.loc['frequencyTo']}
scanDepthFrom\t\t{settings.VALUE.loc['scanDepthFrom']}
scanDepthTo\t\t{settings.VALUE.loc['scanDepthTo']}
verboseFlag\t\t{settings.VALUE.loc['verboseFlag']}
plotSteps1n2Flag\t\t{settings.VALUE.loc['plotSteps1n2Flag']}
"""
        with open(os.path.join(tmpdir, "DSA_SETTINGS.txt"), 'w') as f:
            f.write(settings_content)

        # Copy DSA script with plot-patch preamble injected
        tmp_dsa = os.path.join(tmpdir, "DSA_1.0.3.py")
        with open(str(dsa_script), 'r') as src:
            original = src.read()

        # Inject preamble after the first docstring / imports
        # Find the line after 'import numba as nb' and inject there
        injection_marker = "import numba as nb"
        if injection_marker in original:
            injected = original.replace(
                injection_marker,
                injection_marker + "\n" + _PLOT_PATCH_PREAMBLE
            )
        else:
            # Fallback: inject after matplotlib imports
            injected = original.replace(
                "import matplotlib.pyplot as pltDebug",
                "import matplotlib.pyplot as pltDebug\n" + _PLOT_PATCH_PREAMBLE
            )

        with open(tmp_dsa, 'w') as f:
            f.write(injected)

        plots_dir = os.path.join(tmpdir, "plots")

        if verbose:
            print(f"Running DSA for {example_name}...")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["MPLBACKEND"] = "Agg"
        env["DSA_PLOT_DIR"] = plots_dir

        # Persistent plots directory in results (created AFTER DSA runs)
        _plots_persistent = example_path / "results" / "plots"

        try:
            result = subprocess.run(
                [sys.executable, tmp_dsa],
                cwd=tmpdir,
                capture_output=not verbose,
                text=True,
                timeout=timeout,
                env=env,
            )

            # Collect plots: copy temp plots to persistent dir
            # (must be AFTER DSA script, which rmtree()s the results dir)
            _plots_persistent.mkdir(parents=True, exist_ok=True)
            all_plots = []
            for tp in _collect_plots(plots_dir):
                dest = str(_plots_persistent / os.path.basename(tp))
                shutil.copy2(tp, dest)
                all_plots.append(dest)
            results_plots_dir = example_path / "results"
            if results_plots_dir.exists():
                all_plots.extend(_collect_plots(str(results_plots_dir)))

            # Parse the final depth from stdout
            stdout = result.stdout
            final_depth = None
            prelim_depth = None
            for line in stdout.split('\n'):
                m = re.search(r'finalDepthSolution\s*=\s*([\d.]+)\s*km', line)
                if m:
                    final_depth = float(m.group(1))
                m = re.search(r'prelimSolution=\s*(\d+)\s*km', line)
                if m:
                    prelim_depth = int(m.group(1))

            # Read results CSV
            results_csv = example_path / "results" / "LocatingResults.csv"
            num_stations = 0
            if results_csv.exists():
                try:
                    df = pd.read_csv(str(results_csv))
                    num_stations = len(df)
                except Exception:
                    pass

            return {
                "status": "success",
                "example": example_name,
                "data_path": resolved_data_path,
                "focal_depth_km": final_depth,
                "prelim_depth_km": prelim_depth,
                "num_stations": num_stations,
                "plots": all_plots,
                "results_file": str(results_csv) if results_csv.exists() else None,
                "stdout": stdout[-3000:] if stdout else "",
                "stderr": result.stderr[-1000:] if result.stderr else "",
            }

        except subprocess.TimeoutExpired:
            _plots_persistent.mkdir(parents=True, exist_ok=True)
            all_plots = []
            for tp in _collect_plots(plots_dir):
                dest = str(_plots_persistent / os.path.basename(tp))
                try: shutil.copy2(tp, dest); all_plots.append(dest)
                except: pass
            results_dir = example_path / "results"
            if results_dir.exists():
                all_plots.extend(_collect_plots(str(results_dir)))
            return {
                "status": "timeout",
                "example": example_name,
                "plots": all_plots,
                "error": f"Timed out after {timeout}s",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


def list_dsa_examples() -> Dict:
    """List available DSA examples."""
    _, _, examples_dir, dsa_script = _get_data_dirs()
    examples = {}
    for name, subdir in EXAMPLE_MAP.items():
        path = examples_dir / subdir
        if path.exists():
            sac_count = len(list(path.glob("*.sac")))
            mseed_count = len(list(path.glob("*.mseed")))
            examples[name] = {
                "directory": str(path),
                "has_sac": sac_count > 0,
                "has_mseed": mseed_count > 0,
                "num_stations": max(sac_count, mseed_count) // 3,
            }
    return {
        "dsa_script": str(dsa_script) if dsa_script.exists() else "not found",
        "examples_dir": str(examples_dir),
        "available_examples": examples,
    }


__all__ = ["run_dsa_example", "list_dsa_examples"]
