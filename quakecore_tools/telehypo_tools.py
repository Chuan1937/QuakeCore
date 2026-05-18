"""
TeleHypo (Teleseismic Hypocenter Location) integration for QuakeCore.

Locates teleseismic earthquakes by automatically matching depth phases.
Wraps the original TeleHypo pipeline scripts.

Reference:
    Yuan, J., Ma, H., Yu, J., Liu, Z. & Zhang, S. (2025).
    An approach for teleseismic location by automatically matching depth phase.
    Front. Earth Sci. 13:1539581.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, List


def _get_telehypo_dirs():
    """Get TeleHypo directories."""
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "professional" / "telehypo"
    scripts_dir = data_dir / "scripts"
    examples_dir = data_dir / "examples"
    return repo_root, data_dir, scripts_dir, examples_dir


def _find_telehypo_source_dir() -> Optional[Path]:
    """Try to find the original TeleHypo source directory."""
    candidates = [
        Path("/Users/chuan/Documents/code/QuakeCore-Paper/TeleHypo"),
        Path.home() / "Documents" / "code" / "QuakeCore-Paper" / "TeleHypo",
    ]
    for c in candidates:
        if c.exists() and (c / "0_Run_TeleHypo.py").exists():
            return c
    return None


def _ensure_telehypo_scripts(scripts_dir: Path) -> Dict:
    """Ensure TeleHypo scripts are available in the data directory."""
    scripts_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "0_Run_TeleHypo.py",
        "sub1_FetchData.py",
        "sub2_CalSignalToNoiseRatio.py",
        "sub3_FetchInventory.py",
        "sub4_SelectAzimuthalStations.py",
        "sub5_PreliminaryLocation.py",
        "sub6_PreciseLocation.py",
        "SETTINGS.txt",
    ]
    # Supporting modules
    support_files = [
        "detect_peaks.py", "distance.py", "geodis.py", "getbeta.py",
        "locatePS.py", "picking.py", "psseparation.py", "ssa.py",
        "taupz.py", "woodanderson.py", "calc_mag.py", "stations_plot.py",
    ]

    source_dir = _find_telehypo_source_dir()
    missing = []

    if source_dir:
        # Copy from source
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
        "missing_files": missing if missing else [],
    }


def run_telehypo_example(
    catalog_dir: Optional[str] = None,
    skip_steps: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict:
    """Run TeleHypo pipeline on the provided catalog.

    Args:
        catalog_dir: Path to catalog directory. If None, uses default.
        skip_steps: List of step numbers to skip (1-6)
        verbose: Print detailed output

    Returns:
        Dictionary with results
    """
    _, _, scripts_dir, examples_dir = _get_telehypo_dirs()

    # Ensure scripts are available
    scripts_info = _ensure_telehypo_scripts(scripts_dir)
    if scripts_info["missing_files"]:
        return {
            "error": f"Missing {len(scripts_info['missing_files'])} TeleHypo files. "
                     f"Please copy TeleHypo scripts to {scripts_dir}",
            "missing": scripts_info["missing_files"],
        }

    if catalog_dir is None:
        catalog_dir = str(examples_dir /
                         "catalog_GCMT_2010-03-04_2010-05-13_Mw6.0-8.0_50-300km")

    catalog_path = Path(catalog_dir)
    if not catalog_path.exists():
        return {"error": f"Catalog directory not found: {catalog_dir}"}

    # Create a temp working directory with symlinks/copies of scripts and data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy scripts and SETTINGS.txt
        for item in scripts_dir.iterdir():
            if item.is_file():
                shutil.copy2(str(item), os.path.join(tmpdir, item.name))

        # Create symlink to catalog data
        catalog_link = os.path.join(tmpdir, catalog_path.name)
        if not os.path.exists(catalog_link):
            try:
                os.symlink(str(catalog_path), catalog_link)
            except OSError:
                shutil.copytree(str(catalog_path), catalog_link)

        # Update SETTINGS.txt to point to the catalog
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

        # Determine which steps to run
        all_steps = [
            "sub1_FetchData.py",
            "sub2_CalSignalToNoiseRatio.py",
            "sub3_FetchInventory.py",
            "sub4_SelectAzimuthalStations.py",
            "sub5_PreliminaryLocation.py",
            "sub6_PreciseLocation.py",
        ]

        if skip_steps:
            steps_to_run = [s for i, s in enumerate(all_steps, 1) if i not in skip_steps]
        else:
            steps_to_run = all_steps

        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"

        results = {"steps_completed": [], "steps_failed": [], "steps_skipped": skip_steps or []}

        for step in steps_to_run:
            step_path = os.path.join(tmpdir, step)
            if not os.path.exists(step_path):
                results["steps_skipped"].append(step)
                continue

            if verbose:
                print(f"\n--- Running {step} ---")

            try:
                result = subprocess.run(
                    [sys.executable, step_path],
                    cwd=tmpdir,
                    capture_output=not verbose,
                    text=True,
                    timeout=600,
                    env=env,
                )
                if result.returncode == 0:
                    results["steps_completed"].append(step)
                else:
                    results["steps_failed"].append({
                        "step": step,
                        "returncode": result.returncode,
                        "stderr": result.stderr[-1000:] if result.stderr else "",
                    })
                    if verbose:
                        print(f"  FAILED: {result.stderr[-500:]}")
            except subprocess.TimeoutExpired:
                results["steps_failed"].append({"step": step, "error": "timeout"})
            except Exception as e:
                results["steps_failed"].append({"step": step, "error": str(e)})

        # Find output
        event_dir_pattern = list(catalog_path.glob("20*"))
        output_info = {}
        for ed in event_dir_pattern:
            ed = Path(ed)
            ssnap = ed / "ssnapresults"
            dsa_res = ed / "DSA_results"
            if ssnap.exists():
                output_info["ssnapresults"] = str(ssnap)
            if dsa_res.exists():
                output_info["DSA_results"] = str(dsa_res)

        return {
            "status": "success" if not results["steps_failed"] else "partial",
            "catalog": str(catalog_path),
            "outputs": output_info,
            **results,
        }


def prepare_telehypo_data(
    source_dir: str,
    target_dir: Optional[str] = None,
    symlink: bool = True,
) -> Dict:
    """Prepare TeleHypo data by copying or symlinking from source to QuakeCore data dir.

    Args:
        source_dir: Path to original TeleHypo project (e.g., QuakeCore-Paper/TeleHypo)
        target_dir: Target directory in QuakeCore data. Defaults to data/professional/telehypo/examples
        symlink: If True, create symlink instead of copy

    Returns:
        Status dictionary
    """
    _, data_dir, scripts_dir, examples_dir = _get_telehypo_dirs()

    if target_dir is None:
        target_dir = str(examples_dir)

    source = Path(source_dir)
    if not source.exists():
        return {"error": f"Source directory not found: {source_dir}"}

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    # Copy/symlink scripts
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_files = list(source.glob("*.py")) + list(source.glob("*.txt"))
    for f in script_files:
        dst = scripts_dir / f.name
        if symlink:
            if not dst.exists():
                os.symlink(str(f), str(dst))
        else:
            shutil.copy2(str(f), str(dst))

    # Copy/symlink catalog data
    catalog_src = source / "catalog_GCMT_2010-03-04_2010-05-13_Mw6.0-8.0_50-300km"
    if catalog_src.exists():
        catalog_dst = target / catalog_src.name
        if symlink:
            if not catalog_dst.exists():
                os.symlink(str(catalog_src), str(catalog_dst))
        else:
            if not catalog_dst.exists():
                shutil.copytree(str(catalog_src), str(catalog_dst))

    return {
        "status": "success",
        "scripts_dir": str(scripts_dir),
        "examples_dir": str(examples_dir),
        "method": "symlink" if symlink else "copy",
    }


__all__ = [
    "run_telehypo_example",
    "prepare_telehypo_data",
]
