"""
SeisPolarity integration for QuakeCore.

Polarity picking toolkit using deep learning models.
Wraps the pip-installable `seispolarity` package.

Usage:
    from quakecore_tools.seispolarity_tools import predict_polarity, list_models

Reference:
    SeisPolarity: A comprehensive framework for seismic first-motion polarity picking.
"""

import os
from typing import Dict, List, Optional, Union

import numpy as np


def list_models(details: bool = True) -> Dict:
    """List all available pretrained SeisPolarity models.

    Args:
        details: If True, print detailed model info

    Returns:
        Dictionary of available models and their configurations
    """
    try:
        from seispolarity.inference import Predictor
        return Predictor.list_pretrained(details=details)
    except ImportError:
        return {"error": "seispolarity not installed. Run: pip install seispolarity"}


def predict_polarity(
    waveforms: np.ndarray,
    model_name: str = "ROSS_SCSN",
    device: Optional[str] = None,
    cache_dir: str = "./checkpoints_download",
    force_ud: bool = False,
) -> Dict:
    """Predict P-wave first-motion polarity from waveforms.

    Args:
        waveforms: numpy array of shape (n_samples,) or (batch, n_samples)
                   or (batch, channels, n_samples)
        model_name: Name of pretrained model (e.g., "ROSS_SCSN", "ROSS_GLOBAL",
                    "EQPOLARITY_SCSN", "DITINGMOTION_DITINGSCSN", etc.)
        device: "cuda" or "cpu". If None, auto-detect.
        cache_dir: Directory to store downloaded models
        force_ud: Force output to Up/Down only (no Unknown)

    Returns:
        Dictionary with predictions, probabilities, and metadata
    """
    try:
        from seispolarity.inference import Predictor
    except ImportError:
        return {"error": "seispolarity not installed. Run: pip install seispolarity"}

    try:
        predictor = Predictor(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            force_ud=force_ud,
        )

        # Ensure waveforms are in correct shape
        if waveforms.ndim == 1:
            waveforms = waveforms.reshape(1, -1)

        predictions = predictor.predict(waveforms)

        return {
            "model": model_name,
            "predictions": predictions,
            "num_samples": len(predictions),
        }

    except Exception as e:
        return {"error": str(e)}


def predict_from_stream(
    stream,
    pick_times: List[float],
    model_name: str = "ROSS_SCSN",
    pre_pick_samples: int = 100,
    post_pick_samples: int = 300,
    device: Optional[str] = None,
    cache_dir: str = "./checkpoints_download",
) -> Dict:
    """Predict polarity from ObsPy Stream at specified pick times.

    Args:
        stream: ObsPy Stream object
        pick_times: List of pick times (relative to stream start)
        model_name: Model to use
        pre_pick_samples: Samples before pick
        post_pick_samples: Samples after pick
        device: Device to use
        cache_dir: Model cache directory

    Returns:
        Dictionary with predictions for each pick
    """
    try:
        from seispolarity.inference import Predictor
        from obspy import Stream
    except ImportError as e:
        return {"error": f"Missing dependency: {e}"}

    try:
        predictor = Predictor(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
        )

        results = []
        total_len = pre_pick_samples + post_pick_samples

        for pick_time in pick_times:
            # Extract window around pick
            pick_idx = int(pick_time * stream[0].stats.sampling_rate)
            start_idx = max(0, pick_idx - pre_pick_samples)
            end_idx = min(len(stream[0].data), pick_idx + post_pick_samples)

            # Get Z component data
            z_data = stream.select(component='Z')[0].data[start_idx:end_idx]

            if len(z_data) < total_len:
                # Pad if needed
                z_data = np.pad(z_data, (0, total_len - len(z_data)), mode='constant')

            z_data = z_data[:total_len]
            waveforms = z_data.reshape(1, -1)

            predictions = predictor.predict(waveforms)

            results.append({
                "pick_time": pick_time,
                "prediction": predictions[0] if len(predictions) > 0 else None,
            })

        return {
            "model": model_name,
            "results": results,
            "num_picks": len(results),
        }

    except Exception as e:
        return {"error": str(e)}


def plot_prediction(
    waveform: np.ndarray,
    prediction: Dict,
    output_path: str,
    title: str = "Polarity Prediction",
) -> str:
    """Plot waveform with polarity prediction result.

    Args:
        waveform: 1D numpy array of waveform data
        prediction: Prediction result dict from predict_polarity()
        output_path: Path to save the plot (e.g., 'polarity_result.png')
        title: Plot title

    Returns:
        Path to saved plot file
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(waveform, color='steelblue', linewidth=0.8)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)

        # Annotate prediction
        if "predictions" in prediction:
            pred = prediction["predictions"]
            if isinstance(pred, list) and len(pred) > 0:
                pred = pred[0]
            label = str(pred)
            color = {'Up': 'red', 'Down': 'blue', 'Unknown': 'gray'}.get(label, 'black')
            ax.text(0.02, 0.95, f"Polarity: {label}", transform=ax.transAxes,
                    fontsize=14, fontweight='bold', color=color,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Mark pick position (center of window)
        center = len(waveform) // 2
        ax.axvline(center, color='green', linestyle='--', alpha=0.5, linewidth=1)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_path

    except Exception as e:
        return f"Error generating plot: {e}"


def get_model_info(model_name: str) -> Dict:
    """Get detailed information about a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model details
    """
    try:
        from seispolarity.inference import MODELS_CONFIG
        if model_name in MODELS_CONFIG:
            return MODELS_CONFIG[model_name]
        else:
            return {"error": f"Model '{model_name}' not found. Use list_models() to see available models."}
    except ImportError:
        return {"error": "seispolarity not installed. Run: pip install seispolarity"}


__all__ = [
    "list_models",
    "predict_polarity",
    "predict_from_stream",
    "plot_prediction",
    "get_model_info",
]
