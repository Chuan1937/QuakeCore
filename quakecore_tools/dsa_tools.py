"""
DSA (Depth-Scanning Algorithm) integration for QuakeCore.

Determines focal depth of local/regional earthquakes using 3-component waveforms.

This wrapper runs the original DSA_1.0.3.py script with optimizations:
  - Numba-accelerated cross-correlation (already in original)
  - Settings-driven execution for any example

Reference:
    Yuan, J., Kao, H., & Yu, J. (2020). Depth-Scanning Algorithm. JGR Solid Earth, 125(7).
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional


# Map example names to their data directories and settings
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


def _get_data_dirs():
    """Get the base data directories for DSA."""
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "professional" / "dsa"
    examples_dir = data_dir / "examples"
    dsa_script = data_dir / "DSA_1.0.3.py"
    return repo_root, data_dir, examples_dir, dsa_script


def run_dsa_example(
    example_name: str = "Example1",
    data_dir: Optional[str] = None,
    verbose: bool = False,
) -> Dict:
    """Run DSA on a specified example by executing the original DSA script.

    Args:
        example_name: Example name (e.g., "Example1", "Example2"...)
        data_dir: Override base data directory
        verbose: Print detailed output

    Returns:
        Dictionary with results including output path and status
    """
    repo_root, default_data_dir, examples_dir, dsa_script = _get_data_dirs()

    if data_dir:
        examples_dir = Path(data_dir)

    example_subdir = EXAMPLE_MAP.get(example_name, example_name)
    example_path = examples_dir / example_subdir

    if not example_path.exists():
        return {"error": f"Example directory not found: {example_path}"}

    if not dsa_script.exists():
        return {"error": f"DSA script not found: {dsa_script}. "
                         "Please copy DSA_1.0.3.py to {0}".format(default_data_dir)}

    # Find the settings file for this example
    settings_files = sorted(example_path.parent.glob(f"DSA_SETTINGS_{example_name}*.txt"))
    if not settings_files:
        settings_files = sorted(example_path.glob("DSA_SETTINGS*.txt"))

    if not settings_files:
        return {"error": f"No settings file found for {example_name}"}

    settings_file = settings_files[0]

    # Parse settings to get data path and vel model
    import pandas as pd
    try:
        settings = pd.read_csv(str(settings_file), sep=r'\s+', index_col='PARAMETER')
        orig_data_path = str(settings.VALUE.loc['dataPath'])
        vel_model = str(settings.VALUE.loc['velModel'])
    except Exception as e:
        return {"error": f"Failed to parse settings: {e}"}

    # Resolve data path relative to settings file or to example dir
    if orig_data_path.startswith('./'):
        # Path relative to where DSA is run - use abs path to example
        resolved_data_path = str(example_path) + "/"
    else:
        resolved_data_path = orig_data_path

    if not resolved_data_path.endswith('/'):
        resolved_data_path += '/'

    # Check if waveform files exist
    sac_files = list(example_path.glob("*.sac"))
    mseed_files = list(example_path.glob("*.mseed"))
    if not sac_files and not mseed_files:
        # Check if data is in the resolved path
        sac_files = list(Path(resolved_data_path).glob("*.sac"))
        mseed_files = list(Path(resolved_data_path).glob("*.mseed"))

    if not sac_files and not mseed_files:
        return {"error": f"No waveform files found in {resolved_data_path}"}

    # Verify velocity model exists
    vel_model_file = Path(resolved_data_path) / f"{vel_model}.nd"
    if not vel_model_file.exists():
        return {"error": f"Velocity model not found: {vel_model_file}"}

    # Run DSA: execute the original script with the settings
    # We create a temporary script that sets up path and runs the original
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_settings = os.path.join(tmpdir, "DSA_SETTINGS.txt")

        # Create a settings file pointing to the correct data path
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
        with open(tmp_settings, 'w') as f:
            f.write(settings_content)

        # Copy DSA script to temp dir and run from there
        tmp_dsa = os.path.join(tmpdir, "DSA_1.0.3.py")
        shutil.copy2(str(dsa_script), tmp_dsa)

        if verbose:
            print(f"Running DSA for {example_name}...")
            print(f"  Data: {resolved_data_path}")
            print(f"  Settings: {tmp_settings}")
            print(f"  Script: {tmp_dsa}")

        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
            env["MPLBACKEND"] = "Agg"  # Non-interactive matplotlib

            result = subprocess.run(
                [sys.executable, tmp_dsa],
                cwd=tmpdir,
                capture_output=not verbose,
                text=True,
                timeout=1800,
                env=env,
            )

            if verbose and result.stdout:
                print(result.stdout)
            if verbose and result.stderr:
                print(result.stderr, file=sys.stderr)

            # Find results
            results_path = Path(resolved_data_path) / "results" / "LocatingResults.csv"
            if results_path.exists():
                # Read and summarize results
                results_df = pd.read_csv(str(results_path))
                return {
                    "status": "success",
                    "example": example_name,
                    "data_path": resolved_data_path,
                    "num_stations": len(results_df),
                    "results_file": str(results_path),
                    "stdout": result.stdout[-2000:] if result.stdout else "",
                }
            else:
                return {
                    "status": "completed",
                    "example": example_name,
                    "warning": "No results file found",
                    "stdout": result.stdout[-2000:] if result.stdout else "",
                    "stderr": result.stderr[-2000:] if result.stderr else "",
                }

        except subprocess.TimeoutExpired:
            return {"error": f"DSA timed out after 30 minutes for {example_name}"}
        except Exception as e:
            return {"error": f"DSA execution failed: {e}"}


def list_dsa_examples() -> Dict:
    """List available DSA examples with their data directories."""
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


__all__ = [
    "run_dsa_example",
    "list_dsa_examples",
]
