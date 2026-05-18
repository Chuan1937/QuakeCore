"""
TeleHypo (Teleseismic Hypocenter Location) integration for QuakeCore.

Locates teleseismic earthquakes by automatically matching depth phases.
Wraps the original TeleHypo pipeline and collects plots for frontend display.

Reference:
    Yuan, J., Ma, H., Yu, J., Liu, Z. & Zhang, S. (2025).
    An approach for teleseismic location by automatically matching depth phase.
    Front. Earth Sci. 13:1539581.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

_PLOT_PATCH_PREAMBLE = r'''
# ------------------------------------------------------------
# QuakeCore injection: save all plots to file for frontend display
import os as _qc_os, matplotlib as _qc_mpl
_qc_mpl.use("Agg")
import matplotlib.pyplot as _qc_plt
_QC_PLOT_DIR = _qc_os.environ.get("TELEHYPO_PLOT_DIR", _qc_os.path.join(_qc_os.path.dirname(_qc_os.path.abspath(__file__)), "plots"))
_qc_os.makedirs(_QC_PLOT_DIR, exist_ok=True)
_QC_PLOT_COUNTER = [0]
_QC_SAVE_DPI = 150

def _qc_save_figure(fig=None, prefix="telehypo"):
    _QC_PLOT_COUNTER[0] += 1
    fname = _qc_os.path.join(_QC_PLOT_DIR, f"{prefix}_{_QC_PLOT_COUNTER[0]:04d}.png")
    (fig or _qc_plt.gcf()).savefig(fname, dpi=_QC_SAVE_DPI, bbox_inches="tight")
    print(f"[QuakeCore] Saved figure: {fname}")

import matplotlib.pyplot
matplotlib.pyplot.show = lambda *a, **kw: _qc_save_figure(prefix="telehypo")
try:
    import matplotlib.pyplot as pltDebug
    pltDebug.show = lambda *a, **kw: _qc_save_figure(prefix="telehypo_debug")
except: pass
# ------------------------------------------------------------
'''


def _get_telehypo_dirs():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "professional" / "telehypo"
    scripts_dir = data_dir / "scripts"
    examples_dir = data_dir / "examples"
    return repo_root, data_dir, scripts_dir, examples_dir


def _find_telehypo_source_dir() -> Optional[Path]:
    candidates = [
        Path("/Users/chuan/Documents/code/QuakeCore-Paper/TeleHypo"),
        Path.home() / "Documents" / "code" / "QuakeCore-Paper" / "TeleHypo",
    ]
    for c in candidates:
        if c.exists() and (c / "0_Run_TeleHypo.py").exists():
            return c
    return None


def _ensure_telehypo_scripts(scripts_dir: Path) -> Dict:
    """Ensure TeleHypo scripts are available."""
    scripts_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "0_Run_TeleHypo.py", "SETTINGS.txt",
        "sub1_FetchData.py", "sub2_CalSignalToNoiseRatio.py",
        "sub3_FetchInventory.py", "sub4_SelectAzimuthalStations.py",
        "sub5_PreliminaryLocation.py", "sub6_PreciseLocation.py",
    ]
    support_files = [
        "detect_peaks.py", "distance.py", "geodis.py", "getbeta.py",
        "locatePS.py", "picking.py", "psseparation.py", "ssa.py",
        "taupz.py", "woodanderson.py", "calc_mag.py", "stations_plot.py",
        "tool_plot_brightness function.py", "tool_plot_DSA_depth_solution.py",
        "tool_plot_SNR.py", "tool_plot_solutions_comparison.py",
    ]

    source_dir = _find_telehypo_source_dir()
    missing = []

    if source_dir:
        for f in required_files + support_files:
            src = source_dir / f
            dst = scripts_dir / f
            if src.exists():
                if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                    shutil.copy2(str(src), str(dst))
            else:
                missing.append(f)
    else:
        missing = required_files + support_files

    return {
        "scripts_dir": str(scripts_dir),
        "source_found": source_dir is not None,
        "missing_files": missing,
    }


def _collect_plots(results_dir: str) -> List[str]:
    """Collect all plot files from directory tree."""
    plots = []
    results_path = Path(results_dir)
    if not results_path.exists():
        return plots
    for ext in ('*.png', '*.svg', '*.jpg'):
        for f in sorted(results_path.rglob(ext)):
            plots.append(str(f))
    return plots


def run_telehypo_example(
    catalog_dir: Optional[str] = None,
    skip_steps: Optional[List[int]] = None,
    verbose: bool = False,
    timeout_per_step: int = 600,
) -> Dict:
    """Run TeleHypo pipeline and return results with plot paths.

    Args:
        catalog_dir: Path to catalog directory.
        skip_steps: List of step numbers to skip (1-6).
        verbose: Print detailed output.
        timeout_per_step: Timeout per step in seconds.

    Returns:
        Dict with status, event info, plots[], outputs, step details.
    """
    _, _, scripts_dir, examples_dir = _get_telehypo_dirs()

    scripts_info = _ensure_telehypo_scripts(scripts_dir)
    if scripts_info["missing_files"]:
        return {
            "error": f"Missing {len(scripts_info['missing_files'])} TeleHypo files",
            "missing": scripts_info["missing_files"],
        }

    if catalog_dir is None:
        catalog_dir = str(examples_dir /
                          "catalog_GCMT_2010-03-04_2010-05-13_Mw6.0-8.0_50-300km")

    catalog_path = Path(catalog_dir)
    if not catalog_path.exists():
        return {"error": f"Catalog directory not found: {catalog_dir}"}

    all_steps = [
        "sub1_FetchData.py", "sub2_CalSignalToNoiseRatio.py",
        "sub3_FetchInventory.py", "sub4_SelectAzimuthalStations.py",
        "sub5_PreliminaryLocation.py", "sub6_PreciseLocation.py",
    ]

    if skip_steps:
        steps_to_run = [s for i, s in enumerate(all_steps, 1) if i not in skip_steps]
    else:
        # Auto-detect which steps can be skipped based on existing data
        event_dir = next(catalog_path.glob("20*"), None)
        skip = []
        if event_dir and (event_dir / "ssnapresults").exists():
            # SNR already computed -> skip sub1, sub2
            skip.extend([1, 2])
        if event_dir and (event_dir / "inventory").exists() and list((event_dir / "inventory").glob("*.xml")):
            skip.append(3)
        if event_dir and list(event_dir.glob("*StationWithHighSNR*.csv")):
            skip.append(4)
        steps_to_run = [s for i, s in enumerate(all_steps, 1) if i not in skip]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy scripts
        for item in scripts_dir.iterdir():
            if item.is_file():
                dst = os.path.join(tmpdir, item.name)
                shutil.copy2(str(item), dst)

        # Inject plot-patch into sub6 (the DSA-like precise location step)
        sub6_path = os.path.join(tmpdir, "sub6_PreciseLocation.py")
        if os.path.exists(sub6_path):
            with open(sub6_path, 'r') as f:
                content = f.read()
            if "import numba as nb" in content:
                content = content.replace(
                    "import numba as nb",
                    "import numba as nb\n" + _PLOT_PATCH_PREAMBLE
                )
            else:
                content = content.replace(
                    "import matplotlib.pyplot as pltDebug",
                    "import matplotlib.pyplot as pltDebug\n" + _PLOT_PATCH_PREAMBLE
                )
            # Force plotSteps1n2Flag to 1 to generate plots
            content = content.replace(
                "par12 =  int( SETTINGS.VALUE.loc['plotSteps1n2Flag']  )",
                "par12 = 1  # forced by QuakeCore for plot generation"
            )
            with open(sub6_path, 'w') as f:
                f.write(content)

        # Symlink/copy catalog data
        catalog_link = os.path.join(tmpdir, catalog_path.name)
        if not os.path.exists(catalog_link):
            try:
                os.symlink(str(catalog_path), catalog_link)
            except OSError:
                shutil.copytree(str(catalog_path), catalog_link)

        # Fix settings path
        settings_path = os.path.join(tmpdir, "SETTINGS.txt")
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = f.read()
            settings = settings.replace(
                "./catalog_GCMT_2010-03-04_2010-05-13_Mw6.0-8.0_50-300km/",
                f"./{catalog_path.name}/"
            )
            with open(settings_path, 'w') as f:
                f.write(settings)

        plots_dir = os.path.join(tmpdir, "plots")
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["TELEHYPO_PLOT_DIR"] = plots_dir

        steps_result = {"completed": [], "failed": [], "skipped": skip_steps or []}

        for step in steps_to_run:
            step_path = os.path.join(tmpdir, step)
            if not os.path.exists(step_path):
                steps_result["skipped"].append(step)
                continue
            try:
                result = subprocess.run(
                    [sys.executable, step_path],
                    cwd=tmpdir,
                    capture_output=not verbose,
                    text=True,
                    timeout=timeout_per_step,
                    env=env,
                )
                if result.returncode == 0:
                    steps_result["completed"].append(step)
                else:
                    steps_result["failed"].append({
                        "step": step,
                        "returncode": result.returncode,
                        "stderr": result.stderr[-500:] if result.stderr else "",
                    })
            except subprocess.TimeoutExpired:
                steps_result["failed"].append({"step": step, "error": "timeout"})
            except Exception as e:
                steps_result["failed"].append({"step": step, "error": str(e)})

        # Collect all plots and copy to persistent location under data/
        _, data_dir, _, _ = _get_telehypo_dirs()
        persistent_plots_dir = data_dir / "results" / "plots"
        persistent_plots_dir.mkdir(parents=True, exist_ok=True)

        all_plots = []
        for tp in _collect_plots(plots_dir):
            dest = str(persistent_plots_dir / os.path.basename(tp))
            shutil.copy2(tp, dest)
            all_plots.append(dest)
        # Also collect plots generated in catalog (but only those under data/)
        for p in catalog_path.rglob("*"):
            if p.is_dir():
                for tp in _collect_plots(str(p)):
                    # Copy to persistent location if not already there
                    dest = str(persistent_plots_dir / os.path.basename(tp))
                    if not os.path.exists(dest):
                        shutil.copy2(tp, dest)
                    if dest not in all_plots:
                        all_plots.append(dest)

        # Find output directories
        output_info = {}
        for ed in sorted(catalog_path.glob("20*")):
            ed = Path(ed)
            for sub in ("ssnapresults", "DSA_results", "figures_SNR", "inventory"):
                sub_p = ed / sub
                if sub_p.exists():
                    output_info[sub] = str(sub_p)

        # Try to extract depth from sub5/sub6 output
        final_depth = None
        for p in catalog_path.rglob("LocatingResults.csv"):
            try:
                import pandas as pd
                df = pd.read_csv(p)
                if "Loc(km)" in df.columns:
                    final_depth = float(df["Loc(km)"].median())
                    break
            except Exception:
                pass

        return {
            "status": "success" if not steps_result["failed"] else "partial",
            "catalog": str(catalog_path),
            "focal_depth_km": final_depth,
            "plots": all_plots,
            "outputs": output_info,
            **steps_result,
        }


def run_telehypo_plots(
    event_dir: Optional[str] = None,
    verbose: bool = False,
) -> Dict:
    """Run TeleHypo plotting tools on an existing event result.

    Args:
        event_dir: Path to event directory (e.g., catalog/2010-03-04-22-39-29).
                   If None, auto-detect from default catalog.

    Returns:
        Dict with plot paths.
    """
    _, _, scripts_dir, examples_dir = _get_telehypo_dirs()
    scripts_info = _ensure_telehypo_scripts(scripts_dir)
    if scripts_info["missing_files"]:
        return {"error": f"Missing scripts: {scripts_info['missing_files']}"}

    if event_dir is None:
        catalog = examples_dir / "catalog_GCMT_2010-03-04_2010-05-13_Mw6.0-8.0_50-300km"
        if catalog.exists():
            event_dirs = sorted(catalog.glob("20*"))
            if event_dirs:
                event_dir = str(event_dirs[0])
        if event_dir is None:
            return {"error": "No event directory found. Specify event_dir."}

    event_path = Path(event_dir)
    if not event_path.exists():
        return {"error": f"Event directory not found: {event_dir}"}

    plot_scripts = [
        "tool_plot_SNR.py",
        "tool_plot_brightness function.py",
        "tool_plot_DSA_depth_solution.py",
        "tool_plot_solutions_comparison.py",
    ]

    catalog_path = event_path.parent
    all_plots = []

    for script_name in plot_scripts:
        script_path = scripts_dir / script_name
        if not script_path.exists():
            continue
        try:
            # These scripts read from SETTINGS.txt and process the event
            temp_dir = tempfile.mkdtemp()
            try:
                shutil.copy2(str(script_path), os.path.join(temp_dir, script_name))
                # Also copy SETTINGS if present
                settings_src = scripts_dir / "SETTINGS.txt"
                if settings_src.exists():
                    shutil.copy2(str(settings_src), os.path.join(temp_dir, "SETTINGS.txt"))

                result = subprocess.run(
                    [sys.executable, script_name],
                    cwd=temp_dir,
                    capture_output=not verbose,
                    text=True,
                    timeout=300,
                    env={**os.environ, "MPLBACKEND": "Agg"},
                )
                # Collect any plots generated
                all_plots.extend(_collect_plots(temp_dir))
                # Also check event dir
                all_plots.extend(_collect_plots(str(event_path)))
                all_plots.extend(_collect_plots(str(catalog_path)))

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            if verbose:
                print(f"Plot script {script_name} failed: {e}")

    return {
        "status": "success",
        "plots": list(set(all_plots)),
    }


def prepare_telehypo_data(
    source_dir: str,
    target_dir: Optional[str] = None,
    symlink: bool = True,
) -> Dict:
    """Copy/symlink TeleHypo data to QuakeCore data dir."""
    _, data_dir, scripts_dir, examples_dir = _get_telehypo_dirs()
    if target_dir is None:
        target_dir = str(examples_dir)

    source = Path(source_dir)
    if not source.exists():
        return {"error": f"Source not found: {source_dir}"}

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    for f in list(source.glob("*.py")) + list(source.glob("*.txt")):
        dst = scripts_dir / f.name
        if symlink and not dst.exists():
            os.symlink(str(f), str(dst))
        else:
            shutil.copy2(str(f), str(dst))

    catalog_src = source / "catalog_GCMT_2010-03-04_2010-05-13_Mw6.0-8.0_50-300km"
    if catalog_src.exists():
        catalog_dst = target / catalog_src.name
        if symlink and not catalog_dst.exists():
            os.symlink(str(catalog_src), str(catalog_dst))
        elif not catalog_dst.exists():
            shutil.copytree(str(catalog_src), str(catalog_dst))

    return {
        "status": "success",
        "scripts_dir": str(scripts_dir),
        "examples_dir": str(examples_dir),
    }


__all__ = ["run_telehypo_example", "run_telehypo_plots", "prepare_telehypo_data"]
