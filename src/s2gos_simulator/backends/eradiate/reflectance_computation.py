"""Unified reflectance computation for HDRF, HCRF, and BRF."""

import logging
from typing import Literal, Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

ReflectanceType = Literal["hdrf", "hcrf", "brf", "bhr"]


def align_and_broadcast_datasets(
    radiance: xr.DataArray,
    reference: xr.DataArray,
    wavelength_dim: str = "w",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Align wavelength dimensions and broadcast reference to match radiance shape.

    This handles the common case where radiance and reference datasets may have:
    - Different wavelength grids (interpolates reference to match radiance)
    - Different dimensions (broadcasts reference to match radiance)

    Args:
        radiance: Radiance DataArray (L)
        reference: Reference DataArray (E for irradiance, or similar)
        wavelength_dim: Name of the wavelength dimension (default: "w")

    Returns:
        Tuple of (radiance, aligned_reference) DataArrays
    """
    if wavelength_dim in radiance.dims and wavelength_dim in reference.dims:
        if not np.array_equal(
            radiance[wavelength_dim].values, reference[wavelength_dim].values
        ):
            reference = reference.interp(
                {wavelength_dim: radiance[wavelength_dim]}, method="linear"
            )

    if set(radiance.dims) - set(reference.dims):
        reference = reference.broadcast_like(radiance)

    return radiance, reference


def compute_reflectance_factor(
    radiance: xr.DataArray,
    reference: xr.DataArray,
    reflectance_type: ReflectanceType,
    measurement_id: str,
    cos_sza: Optional[float] = None,
    extra_attrs: Optional[dict] = None,
) -> xr.Dataset:
    """Compute reflectance factor.

    Args:
        radiance: Radiance DataArray (L)
        reference: Reference irradiance DataArray (E)
        reflectance_type: Type of reflectance ("hdrf", "hcrf", or "brf")
        measurement_id: Identifier for this measurement
        cos_sza: Cosine of solar zenith angle (required for BRF)
        extra_attrs: Additional attributes to include in the dataset

    Returns:
        xarray Dataset containing the reflectance factor

    Raises:
        ValueError: If cos_sza not provided for BRF computation
    """
    L, E = align_and_broadcast_datasets(radiance, reference)

    if reflectance_type == "brf":
        if cos_sza is None:
            raise ValueError("cos_sza is required for BRF computation")
        result = (np.pi * L) / (E * cos_sza)
        formula = "BRF = (pi * L) / (E_toa * cos(SZA))"
    elif reflectance_type == "bhr":
        # BHR is simple ratio of radiosity values (no pi factor)
        # L = surface radiosity, E = reference radiosity (white Lambertian disk)
        result = L / E
        formula = "BHR = J_surface / J_reference"
    else:
        result = (np.pi * L) / E
        formula = f"{reflectance_type.upper()} = (pi * L) / E"

    attrs = {
        "measurement_type": reflectance_type,
        "measurement_id": measurement_id,
        "units": "dimensionless",
        "formula": formula,
    }

    if cos_sza is not None:
        attrs["cos_sza"] = float(cos_sza)

    if extra_attrs:
        attrs.update(extra_attrs)

    return xr.Dataset(
        {reflectance_type: result},
        attrs=attrs,
    )
