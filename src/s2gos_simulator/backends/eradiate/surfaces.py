"""Eradiate surface creation functions."""

from pathlib import Path
import numpy as np
from PIL import Image

# Import Eradiate with fallback
try:
    import mitsuba as mi
    MITSUBA_AVAILABLE = True
except ImportError:
    MITSUBA_AVAILABLE = False


def create_target_surface(mesh_file: Path, texture_file: Path) -> dict:
    """Create target surface with material selection.
    
    Args:
        mesh_file: Path to PLY mesh file
        texture_file: Path to selection texture image
        
    Returns:
        Dictionary containing Eradiate surface definitions
    """
    texture_image = Image.open(texture_file)
    selection_texture_data = np.array(texture_image)
    selection_texture_data = np.atleast_3d(selection_texture_data)
    
    material_ids = [
        "_mat_treecover", "_mat_shrubland", "_mat_grassland", "_mat_cropland",
        "_mat_concrete", "_mat_baresoil", "_mat_snow", "_mat_water",
        "_mat_wetland", "_mat_mangroves", "_mat_moss"
    ]
    
    return {
        "terrain_material": {
            "type": "selectbsdf",
            "id": "terrain_material",
            "indices": {
                "type": "bitmap",
                "raw": True,
                "filter_type": "nearest",
                "wrap_mode": "clamp",
                "data": selection_texture_data,
            },
            **{f"bsdf_{i:02d}": {"type": "ref", "id": mat_id}
               for i, mat_id in enumerate(material_ids)}
        },
        "terrain": {
            "type": "ply",
            "filename": str(mesh_file),
            "bsdf": {"type": "ref", "id": "terrain_material"},
            "id": "terrain"
        }
    }


def create_buffer_surface(buffer_mesh_file: Path, buffer_texture_file: Path, mask_file: Path = None) -> dict:
    """Create buffer surface with material selection and optional masking.
    
    Args:
        buffer_mesh_file: Path to buffer PLY mesh file
        buffer_texture_file: Path to buffer selection texture image
        mask_file: Optional path to mask texture for buffer
        
    Returns:
        Dictionary containing Eradiate surface definitions
    """
    buffer_texture_image = Image.open(buffer_texture_file)
    buffer_selection_texture_data = np.array(buffer_texture_image)
    buffer_selection_texture_data = np.atleast_3d(buffer_selection_texture_data)
    
    material_ids = [
        "_mat_treecover", "_mat_shrubland", "_mat_grassland", "_mat_cropland",
        "_mat_concrete", "_mat_baresoil", "_mat_snow", "_mat_water",
        "_mat_wetland", "_mat_mangroves", "_mat_moss"
    ]
    
    result = {
        "buffer_material": {
            "type": "selectbsdf",
            "id": "buffer_material",
            "indices": {
                "type": "bitmap",
                "raw": True,
                "filter_type": "nearest",
                "wrap_mode": "clamp",
                "data": buffer_selection_texture_data,
            },
            **{f"bsdf_{i:02d}": {"type": "ref", "id": mat_id}
               for i, mat_id in enumerate(material_ids)}
        }
    }
    
    buffer_bsdf_id = "buffer_material"
    
    if mask_file and Path(mask_file).exists():
        mask_image = Image.open(mask_file)
        mask_data = np.array(mask_image) / 255.0
        mask_data = np.atleast_3d(mask_data)
        
        result["buffer_mask"] = {
            "type": "mask",
            "opacity": {
                "type": "bitmap",
                "raw": True,
                "filter_type": "nearest",
                "wrap_mode": "clamp",
                "data": mask_data,
            },
            "material": {"type": "ref", "id": "buffer_material"}
        }
        buffer_bsdf_id = "buffer_mask"
    
    result["buffer_terrain"] = {
        "type": "ply",
        "filename": str(buffer_mesh_file),
        "bsdf": {"type": "ref", "id": buffer_bsdf_id},
        "id": "buffer_terrain"
    }
    
    return result


def create_background_surface(elevation: float, mask_texture_path: Path, 
                            material_id: str = "_mat_water", mask_edge_length: float = 100000.0) -> dict:
    """Create background surface using dreams-scenes approach.
    
    Args:
        elevation: Elevation of background surface in meters
        mask_texture_path: Path to background mask texture
        material_id: Material ID for background surface
        mask_edge_length: Edge length of mask in meters
        
    Returns:
        Dictionary containing Eradiate surface definitions
    """
    if not MITSUBA_AVAILABLE:
        raise ImportError("Mitsuba is required for background surface creation")
    
    shape_size = 1e9
    scale = shape_size / mask_edge_length / 3.0
    offset = -0.002
    
    to_world = mi.ScalarTransform4f.translate(
        [0, 0, elevation + offset]
    ) @ mi.ScalarTransform4f.scale(0.5 * shape_size)
    
    to_uv = mi.ScalarTransform4f.scale(
        [scale, scale, 1]
    ) @ mi.ScalarTransform4f.translate(
        [0.5 * (1.0 / scale - 1.0), 0.5 * (1.0 / scale - 1.0), 0.0]
    )
    
    return {
        "background_surface": {
            "type": "rectangle",
            "to_world": to_world,
            "bsdf": {
                "type": "mask",
                "opacity": {
                    "type": "bitmap",
                    "filename": str(mask_texture_path),
                    "raw": True,
                    "filter_type": "nearest",
                    "wrap_mode": "clamp",
                    "to_uv": to_uv,
                },
                "material": {"type": "ref", "id": material_id},
            },
            "id": "background_surface",
        }
    }