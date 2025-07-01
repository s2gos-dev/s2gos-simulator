from typing import List, Union, Dict, Any


def srf_delta(wavelengths: Union[float, List[float]]) -> Dict[str, Any]:
    """Create delta SRF for specific wavelengths.
    
    Args:
        wavelengths: Single wavelength or list of wavelengths in nm
        
    Returns:
        Dictionary for delta SRF configuration
    """
    if isinstance(wavelengths, (int, float)):
        wavelengths = [wavelengths]
    return {"type": "delta", "wavelengths": wavelengths}


def srf_uniform(wavelength_min: float, wavelength_max: float, value: float=1.0) -> Dict[str, Any]:
    """Create uniform SRF across wavelength range.
    
    Args:
        wavelength_min: Minimum wavelength in nm
        wavelength_max: Maximum wavelength in nm
        
    Returns:
        Dictionary for uniform SRF configuration
    """
    return {
        "type": "uniform", 
        "wmin": wavelength_min, 
        "wmax": wavelength_max,
        "value": value
    }


def srf_dataset(dataset_id: str) -> str:
    """Create dataset SRF using string identifier.
    
    Args:
        dataset_id: Dataset identifier like 'sentinel_2a-msi-4'
        
    Returns:
        String identifier for dataset SRF
    """
    return dataset_id


# Convenience functions for common use cases
def srf_rgb() -> Dict[str, Any]:
    """RGB bands (440, 550, 660 nm)."""
    return srf_delta([440.0, 550.0, 660.0])


def srf_visible() -> Dict[str, Any]:
    """Visible spectrum (400-700 nm)."""
    return srf_uniform(400.0, 700.0)


def srf_nir() -> Dict[str, Any]:
    """Near-infrared (700-1000 nm)."""
    return srf_uniform(700.0, 1000.0)


def srf_multispectral(wavelengths: List[float]) -> Dict[str, Any]:
    """Custom multispectral bands."""
    return srf_delta(wavelengths)