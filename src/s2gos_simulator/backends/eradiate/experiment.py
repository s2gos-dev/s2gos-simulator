"""Eradiate experiment creation and assembly."""

from pathlib import Path
from typing import TYPE_CHECKING

# Import with optional dependencies
try:
    import eradiate
    from eradiate.experiments import AtmosphereExperiment
    from eradiate.units import unit_registry as ureg
    ERADIATE_AVAILABLE = True
except ImportError:
    ERADIATE_AVAILABLE = False

# Type imports for annotations
if TYPE_CHECKING:
    pass

from .surfaces import create_target_surface, create_buffer_surface, create_background_surface


def create_experiment(scene_config, render_config, scene_dir: Path):
    """Create an Eradiate experiment from SceneConfig and RenderConfig.
    
    Args:
        scene_config: Scene geometry and materials configuration (from s2gos_generator)
        render_config: Rendering parameters (atmosphere, illumination, sensors)
        scene_dir: Directory containing scene assets
    
    Returns:
        AtmosphereExperiment ready for rendering
        
    Raises:
        ImportError: If Eradiate is not available
    """
    if not ERADIATE_AVAILABLE:
        raise ImportError("Eradiate is not available. Install with: pip install eradiate[kernel]")
    
    # Import here to avoid circular imports (s2gos_generator depends on scene generation)
    try:
        from s2gos_generator.materials.registry import MaterialRegistry
        from s2gos_generator.assets.atmosphere import create_atmosphere
    except ImportError as e:
        raise ImportError(f"s2gos_generator is required for scene materials and atmosphere: {e}")
    
    # Create materials
    materials = scene_config.materials
    kdict, kpmap = MaterialRegistry.create_material_kdict_kpmap(materials)
    
    # Create target surface
    target_mesh_path = scene_dir / scene_config.target["mesh"]
    target_texture_path = scene_dir / scene_config.target["selection_texture"]
    kdict.update(create_target_surface(target_mesh_path, target_texture_path))
    
    # Create buffer surface if configured
    if scene_config.buffer:
        buffer_mesh_path = scene_dir / scene_config.buffer["mesh"]
        buffer_texture_path = scene_dir / scene_config.buffer["selection_texture"]
        mask_path = scene_dir / scene_config.buffer["mask_texture"] if "mask_texture" in scene_config.buffer else None
        kdict.update(create_buffer_surface(buffer_mesh_path, buffer_texture_path, mask_path))
    
    # Create background surface if configured
    if scene_config.background:
        bg_mask_path = scene_dir / scene_config.background["mask_texture"]
        kdict.update(create_background_surface(
            elevation=scene_config.background["elevation"],
            mask_texture_path=bg_mask_path,
            material_id="_mat_water",
            mask_edge_length=scene_config.background["mask_edge_length"]
        ))
        print(f"Added background surface at elevation {scene_config.background['elevation']:.1f}m")
    
    # Create atmosphere from scene config (tied to geographic location)
    atmosphere = create_atmosphere(
        boa=scene_config.atmosphere.get("boa", 0.0),
        toa=scene_config.atmosphere.get("toa", 40.0e3),
        aerosol_ot=scene_config.atmosphere.get("aerosol_ot", 0.1),
        aerosol_scale=scene_config.atmosphere.get("aerosol_scale", 1e3),
        aerosol_ds=scene_config.atmosphere.get("aerosol_ds", "sixsv-continental")
    )
    
    # Create illumination from render config
    illumination = {
        "type": "directional",
        "zenith": render_config.illumination.zenith * ureg.deg,
        "azimuth": render_config.illumination.azimuth * ureg.deg,
        "irradiance": {
            "type": render_config.illumination.irradiance_type,
            "dataset": render_config.illumination.irradiance_dataset
        }
    }
    
    # Create measures from render config sensors
    measures = []
    for sensor in render_config.sensors:
        measures.append({
            "type": sensor.type,
            "id": sensor.id,
            "origin": sensor.origin,
            "target": sensor.target,
            "up": [0, 1, 0],
            "fov": sensor.fov,
            "film_resolution": tuple(sensor.resolution),
            "srf": {
                "type": "delta",
                "wavelengths": [440, 550, 660] * ureg.nm
            },
            "spp": sensor.spp
        })
    
    return AtmosphereExperiment(
        geometry={"type": "plane_parallel", "toa_altitude": 40.0 * ureg.km},
        atmosphere=atmosphere,
        surface=None,
        illumination=illumination,
        measures=measures,
        kdict=kdict,
        kpmap=kpmap
    )