"""Reflectance factor computation for remote sensing simulations.

This module provides automated computation of reflectance factors from
radiance and irradiance measurements, critical for HYPSTAR and other
ground-based sensor validation workflows.

The reflectance factor is computed as:
    ρ = π × L / E

where:
    L = measured radiance (W/m²/sr/nm)
    E = BOA downward irradiance (W/m²/nm)
    ρ = dimensionless reflectance factor
"""

import logging
from typing import Dict

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class ReflectanceProcessor:
    """Processes radiance and irradiance data to compute reflectance factors.

    This class automates the calculation of reflectance factors from
    simulation results, handling coordinate alignment, dimension matching,
    and metadata propagation.
    """

    def compute_reflectance_factor(
        self,
        radiance: xr.DataArray,
        irradiance: xr.DataArray,
        sensor_id: str = None,
    ) -> xr.DataArray:
        """Compute reflectance factor from radiance and irradiance.

        The reflectance factor is computed as:
            ρ = π × L / E

        This is the hemispherical-directional reflectance factor (HDRF) when
        measuring upward radiance and downward hemispherical irradiance.

        Args:
            radiance: Measured radiance DataArray (W/m²/sr/nm)
            irradiance: BOA downward irradiance DataArray (W/m²/nm)
            sensor_id: Optional sensor identifier for metadata

        Returns:
            Reflectance factor DataArray (dimensionless)
        """
        # Align wavelength coordinates if needed
        if "w" in radiance.dims and "w" in irradiance.dims:
            # Interpolate irradiance to match radiance wavelengths
            irradiance_aligned = irradiance.interp(w=radiance.coords["w"])
        else:
            irradiance_aligned = irradiance

        # Compute reflectance factor: ρ = π × L / E
        reflectance = (np.pi * radiance) / irradiance_aligned

        # Add metadata
        reflectance.attrs.update(
            {
                "quantity": "reflectance_factor",
                "long_name": "Hemispherical-directional reflectance factor",
                "units": "dimensionless",
                "formula": "ρ = π × L / E",
                "radiance_units": "W m^-2 sr^-1 nm^-1",
                "irradiance_units": "W m^-2 nm^-1",
            }
        )

        if sensor_id is not None:
            reflectance.attrs["sensor_id"] = sensor_id

        logger.info(
            f"Reflectance factor computed: "
            f"mean={float(reflectance.mean()):.4f}, "
            f"min={float(reflectance.min()):.4f}, "
            f"max={float(reflectance.max()):.4f}"
        )

        return reflectance

    def add_reflectance_to_results(
        self,
        sensor_results: Dict[str, xr.Dataset],
        irradiance_results: Dict[str, xr.Dataset],
        sensor_to_irradiance_mapping: Dict[str, str] = None,
    ) -> Dict[str, xr.Dataset]:
        """Add reflectance factor to sensor results using irradiance data.

        This method augments sensor datasets with computed reflectance factors,
        creating a unified result structure with radiance, irradiance, and
        reflectance all in one dataset.

        Args:
            sensor_results: Dictionary of sensor datasets containing 'radiance'
            irradiance_results: Dictionary of irradiance datasets
            sensor_to_irradiance_mapping: Optional mapping of sensor IDs to
                irradiance measurement IDs. If None, assumes matching IDs or
                uses the first irradiance measurement for each sensor.

        Returns:
            Updated sensor results with reflectance factors added
        """
        # Default mapping: assume matching IDs or use first irradiance
        if sensor_to_irradiance_mapping is None:
            if len(irradiance_results) == 1:
                # Single irradiance measurement - use for all sensors
                irr_id = list(irradiance_results.keys())[0]
                sensor_to_irradiance_mapping = {
                    sensor_id: irr_id for sensor_id in sensor_results.keys()
                }
            else:
                # Assume matching IDs
                sensor_to_irradiance_mapping = {
                    sensor_id: sensor_id for sensor_id in sensor_results.keys()
                }

        updated_results = {}

        for sensor_id, sensor_ds in sensor_results.items():
            # Skip if this sensor doesn't have corresponding irradiance
            if sensor_id not in sensor_to_irradiance_mapping:
                logger.warning(
                    f"No irradiance mapping for sensor '{sensor_id}' - skipping reflectance"
                )
                updated_results[sensor_id] = sensor_ds
                continue

            irr_id = sensor_to_irradiance_mapping[sensor_id]
            if irr_id not in irradiance_results:
                logger.warning(
                    f"Irradiance measurement '{irr_id}' not found for sensor '{sensor_id}'"
                )
                updated_results[sensor_id] = sensor_ds
                continue

            # Get radiance and irradiance
            if "radiance" not in sensor_ds:
                logger.warning(
                    f"No 'radiance' variable in sensor '{sensor_id}' dataset"
                )
                updated_results[sensor_id] = sensor_ds
                continue

            radiance = sensor_ds["radiance"]
            irradiance = irradiance_results[irr_id]["irradiance"]

            # Compute reflectance
            logger.info(
                f"Computing reflectance for sensor '{sensor_id}' using "
                f"irradiance '{irr_id}'"
            )
            reflectance = self.compute_reflectance_factor(
                radiance, irradiance, sensor_id=sensor_id
            )

            # Create updated dataset with all variables
            updated_ds = sensor_ds.copy()
            updated_ds["reflectance_factor"] = reflectance
            updated_ds["boa_irradiance"] = irradiance

            # Add metadata about the computation
            updated_ds.attrs["reflectance_computed"] = True
            updated_ds.attrs["irradiance_source"] = irr_id

            updated_results[sensor_id] = updated_ds

        logger.info(
            f"Reflectance factors added to {len(updated_results)} sensor results"
        )

        return updated_results

    def compute_uncertainty(
        self,
        radiance: xr.DataArray,
        irradiance: xr.DataArray,
        radiance_std: xr.DataArray = None,
        irradiance_std: xr.DataArray = None,
    ) -> xr.DataArray:
        """Compute reflectance factor uncertainty via error propagation.

        Uses standard error propagation for division:
            σ_ρ/ρ = sqrt((σ_L/L)² + (σ_E/E)²)

        Args:
            radiance: Radiance measurements
            irradiance: Irradiance measurements
            radiance_std: Optional radiance standard deviation (Monte Carlo noise)
            irradiance_std: Optional irradiance standard deviation

        Returns:
            Reflectance factor uncertainty (standard deviation)
        """
        if radiance_std is None or irradiance_std is None:
            logger.warning(
                "Standard deviations not provided - cannot compute uncertainty"
            )
            return None

        # Relative uncertainties
        rel_rad_unc = radiance_std / radiance
        rel_irr_unc = irradiance_std / irradiance

        # Combined relative uncertainty
        rel_reflectance_unc = np.sqrt(rel_rad_unc**2 + rel_irr_unc**2)

        # Absolute uncertainty
        reflectance = (np.pi * radiance) / irradiance
        reflectance_unc = reflectance * rel_reflectance_unc

        reflectance_unc.attrs.update(
            {
                "quantity": "reflectance_factor_uncertainty",
                "long_name": "Reflectance factor standard deviation",
                "units": "dimensionless",
                "method": "error_propagation",
            }
        )

        return reflectance_unc
