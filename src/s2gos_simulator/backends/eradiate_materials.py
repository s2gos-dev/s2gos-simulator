"""Eradiate-specific material functionality for S2GOS simulator backend."""

import logging
from upath import UPath
from typing import Callable, Dict

import xarray as xr

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
    spectral_dict: Dict[str, str],
) -> Callable[[KernelContext], float]:
    """Create callable function from spectral parameter dictionary.

    Args:
        spectral_dict: Dictionary with 'path' and 'variable' keys

    Returns:
        Callable that evaluates spectral data for given KernelContext
    """
    if not ERADIATE_AVAILABLE:
        # Fallback when Eradiate is not available
        def dummy_func(ctx) -> float:
            return 0.5

        return dummy_func

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

        def spectral_func(ctx: KernelContext) -> float:
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

        def fallback_func(ctx) -> float:
            return 0.5

        return fallback_func


def _declare_mono_scene_parameter(
    func: Callable[[KernelContext], float], node_id: str, param: str
) -> UpdateParameter:
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
    def load_spectral_data(file_path: UPath, variable: str = "reflectance"):
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
    def create_ocean_kpmap(material) -> dict:
        """Generate Eradiate parameter map for ocean material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        return {}  # Ocean materials typically don't have spectral parameters
