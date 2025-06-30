"""Eradiate rendering execution and result processing."""

from pathlib import Path
import numpy as np
from PIL import Image

# Import with optional dependencies
try:
    import eradiate
    from eradiate.xarray.interp import dataarray_to_rgb
    ERADIATE_AVAILABLE = True
except ImportError:
    ERADIATE_AVAILABLE = False

from .experiment import create_experiment


def render_experiment(scene_config, render_config, scene_dir: Path, output_dir: Path = None):
    """Render scene using Eradiate for physically-based results.
    
    Args:
        scene_config: Scene geometry and materials configuration
        render_config: Rendering parameters (atmosphere, illumination, sensors)
        scene_dir: Directory containing scene assets (meshes, textures, etc.)
        output_dir: Directory for render outputs (defaults to scene_dir/eradiate_renders)
    
    Returns:
        dict: Rendering results with paths to output files
        
    Raises:
        ImportError: If Eradiate is not available
        RuntimeError: If rendering fails
    """
    if not ERADIATE_AVAILABLE:
        raise ImportError("Eradiate is not available. Install with: pip install eradiate[kernel]")
    
    if output_dir is None:
        output_dir = scene_dir / "eradiate_renders"
        
    print(f"Rendering scene '{scene_config.name}' with Eradiate...")
    print(f"Scene location: {scene_config.metadata.center_lat}, {scene_config.metadata.center_lon}")
    print(f"Target: {scene_config.target['mesh']} + {scene_config.target['selection_texture']}")
    if scene_config.buffer:
        print(f"Buffer: {scene_config.buffer['mesh']} + {scene_config.buffer['selection_texture']}")
    if scene_config.background:
        print(f"Background: {scene_config.background['material']} at {scene_config.background['elevation']:.1f}m")
    
    try:
        # Create the scene experiment
        exp = create_experiment(scene_config, render_config, scene_dir)
        
        print("Running Eradiate simulation (this may take several minutes)...")
        eradiate.run(exp)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        raw_output = output_dir / "eradiate_results.nc"
        results_ds = exp.results["perspective_view"]
        results_ds.to_netcdf(raw_output)
        
        # Create RGB visualization
        img = dataarray_to_rgb(
            results_ds["radiance"],
            channels=[("w", 660), ("w", 550), ("w", 440)],
            normalize=False,
        ) * 1.8
        
        img = np.clip(img, 0, 1)
        
        # Save RGB image
        rgb_output = output_dir / "eradiate_rgb.png"
        plt_img = (img * 255).astype(np.uint8)
        rgb_image = Image.fromarray(plt_img)
        rgb_image.save(rgb_output)
        
        print(f"Eradiate simulation complete!")
        print(f"Raw results: {raw_output}")
        print(f"RGB visualization: {rgb_output}")
        
        return {
            "success": True,
            "raw_results": raw_output,
            "rgb_image": rgb_output,
            "output_dir": output_dir,
            "experiment": exp
        }
        
    except Exception as e:
        print(f"Eradiate rendering failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "output_dir": output_dir if 'output_dir' in locals() else None
        }