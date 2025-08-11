"""Eradiate-specific material functionality for S2GOS simulator backend."""

import logging
from typing import Any, Callable, Dict, Union

import numpy as np
import xarray as xr
from upath import UPath

from eradiate import KernelContext
from eradiate.kernel import UpdateParameter

try:
    import mitsuba as mi
    from eradiate import KernelContext
    from eradiate.kernel import TypeIdLookupStrategy, UpdateParameter
    from eradiate.scenes.spectra import InterpolatedSpectrum

    ERADIATE_AVAILABLE = True
except ImportError:
    KernelContext = None
    InterpolatedSpectrum = None
    UpdateParameter = None
    TypeIdLookupStrategy = None
    mi = None
    ERADIATE_AVAILABLE = False


def _create_spectral_callable(
    spectral_dict: Dict[str, Any],
) -> Callable[["KernelContext"], float]:
    """Create callable function from spectral parameter dictionary.

    Supports both spectral file references and uniform values:
    - File reference: {"path": "spectrum.nc", "variable": "reflectance"}
    - Uniform value: {"type": "uniform", "value": 0.5}

    Args:
        spectral_dict: Dictionary with spectral data specification

    Returns:
        Callable that evaluates spectral data for given KernelContext
    """
    if not ERADIATE_AVAILABLE:
        # Fallback when Eradiate is not available
        def dummy_func(_ctx) -> float:
            return 0.5

        return dummy_func

    if spectral_dict.get("type") == "uniform":
        uniform_value = spectral_dict["value"]
        
        if isinstance(uniform_value, (int, float)):
            def uniform_scalar_func(_ctx) -> float:
                return float(uniform_value)
                
            logging.info(f"Using uniform scalar value: {uniform_value}")
            return uniform_scalar_func
            
        elif isinstance(uniform_value, (list, tuple)) and len(uniform_value) == 3:
            r, g, b = uniform_value
            
            def uniform_rgb_func(ctx: "KernelContext") -> float:
                wavelength = ctx.si.wavelength.m_as("nanometer") if hasattr(ctx.si, 'wavelength') else 550.0
                
                if wavelength <= 440:
                    return float(b)
                elif wavelength <= 540:  # Green-ish  
                    return float(g)
                else:
                    return float(r)
                    
            logging.info(f"Using uniform RGB value: {uniform_value}")
            return uniform_rgb_func
            
        else:
            logging.warning(f"Invalid uniform value format: {uniform_value}, using fallback")
            def fallback_uniform_func(_ctx) -> float:
                return 0.5
            return fallback_uniform_func
    
    if "path" not in spectral_dict or "variable" not in spectral_dict:
        logging.error(f"Invalid spectral dictionary format: {spectral_dict}")
        def fallback_func(_ctx) -> float:
            return 0.5
        return fallback_func
        
    file_path = spectral_dict["path"]
    variable = spectral_dict["variable"]

    # Resolve the path relative to generator data directory
    if not UPath(file_path).is_absolute():
        # Assume relative to s2gos-generator data directory
        import s2gos_generator

        generator_dir = UPath(s2gos_generator.__file__).parent
        full_path = generator_dir / "data" / file_path
    else:
        full_path = UPath(file_path)

    # Load spectral data using Eradiate
    try:
        # Check if file exists before trying to load
        from s2gos_utils.io.paths import exists

        if not exists(full_path):
            raise FileNotFoundError(f"Spectral data file not found: {full_path}")

        # Load the spectral data directly with xarray
        dataset = xr.load_dataset(full_path)
        if variable not in dataset.data_vars:
            raise KeyError(
                f"Variable '{variable}' not found in dataset. Available: {list(dataset.data_vars.keys())}"
            )

        da = dataset[variable]
        spectrum = InterpolatedSpectrum.from_dataarray(dataarray=da)

        def spectral_func(ctx: "KernelContext") -> float:
            return spectrum.eval(ctx.si).m_as("dimensionless")

        logging.info(
            f"Successfully loaded spectral data: {full_path} (variable: {variable})"
        )
        return spectral_func

    except Exception as e:
        # Log the error and fallback
        logging.error(
            f"Failed to load spectral data from {full_path} (variable: {variable}): {e}"
        )
        logging.warning("Using fallback constant value 0.5 for spectral parameter")

        def fallback_func(_ctx) -> float:
            return 0.5

        return fallback_func


def _declare_mono_scene_parameter(
    func: Callable[["KernelContext"], float], node_id: str, param: str
) -> "UpdateParameter":
    """Create UpdateParameter for monochromatic mode.

    Args:
        func: Function that computes parameter value from kernel context
        node_id: Eradiate node identifier
        param: Parameter path relative to the node

    Returns:
        UpdateParameter configured for spectral evaluation
    """
    if not ERADIATE_AVAILABLE:
        raise ImportError("Eradiate is not available")

    return UpdateParameter(
        func,
        lookup_strategy=TypeIdLookupStrategy(
            node_type=mi.BSDF, node_id=node_id, parameter_relpath=param
        ),
        flags=UpdateParameter.Flags.SPECTRAL,
    )


class EradiateMaterialAdapter:
    """Adapter for converting S2GOS materials to Eradiate kernel dictionaries."""

    @staticmethod
    def load_spectral_data(file_path: UPath, variable: str = "reflectance") -> Any:
        """Load spectral data from file using Eradiate's InterpolatedSpectrum."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        import xarray as xr

        data = xr.open_dataset(file_path)
        return InterpolatedSpectrum.from_dataarray(dataarray=data[variable])

    @staticmethod
    def create_diffuse_kdict(material) -> dict:
        """Generate Eradiate kernel dictionary for diffuse material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        # Setting the initial reflectance to 0 can help detect parameter update issues
        result = {
            material.mat_id: {
                "type": "diffuse",
                "id": material.mat_id,
                "reflectance": 0.0,
            }
        }

        return result

    @staticmethod
    def create_diffuse_kpmap(material) -> dict:
        """Generate Eradiate parameter map for diffuse material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        result = {}

        node_id = material.mat_id
        param = "reflectance.value"
        # Convert dict to callable function
        reflectance_func = _create_spectral_callable(material.reflectance)
        result[f"{node_id}.{param}"] = _declare_mono_scene_parameter(
            reflectance_func, node_id=node_id, param=param
        )

        return result

    @staticmethod
    def create_bilambertian_kdict(material) -> dict:
        """Generate Eradiate kernel dictionary for bilambertian material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        result = {material.mat_id: {"type": "bilambertian", "id": material.mat_id}}

        return result

    @staticmethod
    def create_bilambertian_kpmap(material) -> dict:
        """Generate Eradiate parameter map for bilambertian material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        result = {}

        node_id = material.mat_id
        for param, spectral_dict in [
            ("reflectance.value", material.reflectance),
            ("transmittance.value", material.transmittance),
        ]:
            # Convert dict to callable function
            spectral_func = _create_spectral_callable(spectral_dict)
            result[f"{node_id}.{param}"] = _declare_mono_scene_parameter(
                spectral_func, node_id=node_id, param=param
            )

        return result

    @staticmethod
    def create_rpv_kdict(material) -> dict:
        """Generate Eradiate kernel dictionary for RPV material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        # Important: to ensure that we use the 4-parameter RPV BRDF, we must
        # pass a value for `rho_c`
        result = {material.mat_id: {"type": "rpv", "id": material.mat_id, "rho_c": 0.5}}

        return result

    @staticmethod
    def create_rpv_kpmap(material) -> dict:
        """Generate Eradiate parameter map for RPV material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        result = {}

        node_id = material.mat_id
        for param, spectral_dict in [
            ("rho_0.value", material.rho_0),
            ("k.value", material.k),
            ("g.value", material.Theta),
            ("rho_c.value", material.rho_c),
        ]:
            # Convert dict to callable function
            spectral_func = _create_spectral_callable(spectral_dict)
            result[f"{node_id}.{param}"] = _declare_mono_scene_parameter(
                spectral_func, node_id=node_id, param=param
            )

        return result

    @staticmethod
    def create_ocean_kdict(material) -> dict:
        """Generate Eradiate kernel dictionary for ocean material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        kdict = {
            "type": "ocean_legacy",
            "id": material.mat_id,
            "wavelength": 550.0,  # Default wavelength for ocean legacy material
            "chlorinity": getattr(material, "chlorinity", 0.0),
            "pigmentation": getattr(material, "pigmentation", 5.0),
            "wind_speed": getattr(material, "wind_speed", 2.0),
            "wind_direction": getattr(material, "wind_direction", 90.0),
        }
        return {material.mat_id: kdict}

    @staticmethod
    def create_ocean_kpmap(_material) -> dict:
        """Generate Eradiate parameter map for ocean material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        return {}  # Ocean materials typically don't have spectral parameters

    @staticmethod
    def create_hamster_kdict(
        material_id: str, 
        albedo_data: xr.DataArray
    ) -> dict:
        """Generate Eradiate kernel dictionary for HAMSTER albedo material.
        
        Args:
            material_id: Material identifier (e.g., "_mat_baresoil_hamster")
            albedo_data: HAMSTER albedo DataArray with (lat, lon, wavelength) dimensions
            
        Returns:
            Dictionary with texture and material definitions for HAMSTER albedo
        """
        texture_data = np.ones_like(albedo_data.values[:, :, :3])
        texture_id = f"texture_{material_id}"
        
        result = {
            texture_id: {
                "type": "bitmap",
                "id": texture_id,
                "filter_type": "nearest", 
                "wrap_mode": "clamp",
                "data": texture_data,
                "raw": True,
            },
            material_id: {
                "type": "diffuse",
                "id": material_id,
                "reflectance": {"type": "ref", "id": texture_id},
            }
        }
        
        return result

    @staticmethod
    def create_hamster_kpmap(
        material_id: str,
        albedo_data: xr.DataArray
    ) -> dict:
        """Generate Eradiate parameter map for HAMSTER albedo material.
        
        Args:
            material_id: Material identifier 
            albedo_data: HAMSTER albedo DataArray with spectral interpolation support
            
        Returns:
            Dictionary with scene parameters for spectral interpolation
        """
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        def hamster_spectral_func(ctx: "KernelContext") -> np.ndarray:
            """Interpolate HAMSTER albedo data for given wavelength."""
            interpolated = albedo_data.sel(wavelength=ctx.si.w, method="nearest")
            return interpolated.values[..., np.newaxis]

        param_key = f"{material_id}.reflectance.data"
        
        result = {
            param_key: UpdateParameter(
                hamster_spectral_func,
                lookup_strategy=TypeIdLookupStrategy(
                    node_type=mi.BSDF,
                    node_id=material_id,
                    parameter_relpath="reflectance.data"
                ),
                flags=UpdateParameter.Flags.SPECTRAL,
            )
        }
        
        return result
