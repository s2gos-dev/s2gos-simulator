import logging
from typing import Dict, List, Optional, Tuple

import xarray as xr
from s2gos_utils.scene import SceneDescription
from upath import UPath

from .irradiance_processor import IrradianceProcessor

logger = logging.getLogger(__name__)


class HDRFProcessor:
    """Processor for HDRF measurements: HDRF = L_actual / L_reference."""

    def __init__(self, backend):
        self.backend = backend
        self.simulation_config = backend.simulation_config
        self.irradiance_processor = IrradianceProcessor(backend)

    def requires_hdrf(self) -> bool:
        """Check if any measurements require HDRF computation.

        Returns:
            True if HDRF computation is needed
        """
        from .config import HDRFConfig

        # Check if any sensor produces HDRF
        for sensor in self.simulation_config.sensors:
            if hasattr(sensor, "produces") and "hdrf" in sensor.produces:
                return True

        # Check if any measurement is an HDRF config
        for measurement in self.simulation_config.measurements:
            if isinstance(measurement, HDRFConfig):
                return True

        return False

    def get_hdrf_measure_ids(self) -> List[str]:
        """Get list of measure IDs that produce HDRF.

        Returns:
            List of measure IDs that require HDRF computation
        """
        from .config import HDRFConfig

        hdrf_ids = []

        # Check sensors that produce HDRF
        for sensor in self.simulation_config.sensors:
            if hasattr(sensor, "produces") and "hdrf" in sensor.produces:
                hdrf_ids.append(sensor.id)

        # Check HDRF measurement configs
        for measurement in self.simulation_config.measurements:
            if isinstance(measurement, HDRFConfig):
                hdrf_id = self._get_hdrf_id(measurement)
                hdrf_ids.append(hdrf_id)

        return [x.replace(".", "_") for x in hdrf_ids]

    def _get_hdrf_id(self, measurement) -> str:
        """Generate ID for an HDRF measurement.

        Args:
            measurement: HDRFConfig instance

        Returns:
            ID string for the measurement
        """
        if measurement.id:
            return measurement.id

        # Auto-generate ID from instrument type
        return f"hdrf_{measurement.instrument}"

    def _get_hdrf_config_for_measure(
        self, measure_id: str
    ) -> Optional[Tuple[float, float, float]]:
        """Get target location and height offset for an HDRF measure.

        Args:
            measure_id: HDRF measure ID

        Returns:
            Tuple of (target_lat, target_lon, height_offset_m) or None if not found.
            Returns (None, None, height_offset) if location uses scene coordinates
            instead of lat/lon.
        """
        from .config import HDRFConfig

        for measurement in self.simulation_config.measurements:
            if isinstance(measurement, HDRFConfig):
                hdrf_id = self._get_hdrf_id(measurement)

                if hdrf_id.replace(".", "_") == measure_id:
                    # Extract location based on instrument type
                    target_lat = None
                    target_lon = None

                    if (
                        measurement.instrument == "hemispherical"
                        and measurement.location
                    ):
                        # Hemispherical HDRF uses location-based pattern
                        target_lat = measurement.location.target_lat
                        target_lon = measurement.location.target_lon
                    elif measurement.viewing:
                        # Radiancemeter uses viewing geometry - try to get target lat/lon
                        # Note: viewing.target is in scene coordinates, not lat/lon
                        # We'll return None and let caller fall back to scene center
                        pass

                    return (
                        target_lat,
                        target_lon,
                        measurement.reference_height_offset_m,
                    )

        return None

    def create_white_reference_scene(
        self, scene_description: SceneDescription, scene_dir: UPath
    ) -> Tuple[SceneDescription, Tuple[float, float, float]]:
        """Create white reference scene for HDRF (delegates to IrradianceProcessor)."""
        hdrf_measure_ids = self.get_hdrf_measure_ids()
        if not hdrf_measure_ids:
            raise ValueError("No HDRF measures found")

        hdrf_config = self._get_hdrf_config_for_measure(hdrf_measure_ids[0])
        if hdrf_config is None:
            raise ValueError(f"HDRF config not found for '{hdrf_measure_ids[0]}'")

        lat, lon, height_offset = hdrf_config

        # Use scene center if no specific location configured
        if lat is None or lon is None:
            lat = scene_description.location["center_lat"]
            lon = scene_description.location["center_lon"]

        # Delegate to IrradianceProcessor
        return self.irradiance_processor.create_reference_disk_scene(
            scene_description,
            scene_dir,
            lat,
            lon,
            height_offset,
            disk_id="white_reference_disk_hdrf",
        )

    def compute_hdrf_from_radiance_and_irradiance(
        self,
        radiance_results: Dict[str, xr.Dataset],
        irradiance_results: Dict[str, xr.Dataset],
        output_dir: UPath,
    ) -> Dict[str, xr.Dataset]:
        """Compute HDRF = (π × L_actual) / E_reference.

        This is the CORRECT approach: use measured radiance and irradiance.
        NO dual simulation needed - that duplicates irradiance measurement!

        Args:
            radiance_results: Actual scene radiance measurements (L_actual)
            irradiance_results: Irradiance measurements (E_reference)
            output_dir: Output directory for HDRF results

        Returns:
            Dictionary of HDRF datasets by measure ID
        """
        import numpy as np
        from s2gos_utils.io.paths import mkdir

        from .config import HDRFConfig

        logger.info("Computing HDRF from radiance + irradiance...")

        mkdir(output_dir)
        hdrf_datasets = {}

        # Get HDRF configs
        hdrf_configs = [
            m for m in self.simulation_config.measurements if isinstance(m, HDRFConfig)
        ]

        for hdrf_config in hdrf_configs:
            hdrf_id = (
                hdrf_config.id
                or f"hdrf_{hdrf_config.viewing_zenith}_{hdrf_config.viewing_azimuth}"
            )
            ref_id = hdrf_config.irradiance_measurement_id

            logger.debug(f"Processing HDRF '{hdrf_id}' with reference '{ref_id}'")

            # Get radiance from actual scene
            if hdrf_id not in radiance_results:
                logger.warning(f"  Skipping: no radiance results for {hdrf_id}")
                continue

            radiance_ds = radiance_results[hdrf_id]
            if "radiance" not in radiance_ds:
                logger.warning(f"  Skipping: no radiance variable in {hdrf_id}")
                continue

            L_actual = radiance_ds["radiance"]

            # Get irradiance from reference measurement
            if ref_id not in irradiance_results:
                logger.error(f"  ERROR: Reference irradiance '{ref_id}' not found!")
                logger.error(f"  Available: {list(irradiance_results.keys())}")
                continue

            irradiance_ds = irradiance_results[ref_id]
            if "boa_irradiance" not in irradiance_ds:
                logger.warning(f"  Skipping: no boa_irradiance in {ref_id}")
                continue

            E_reference = irradiance_ds["boa_irradiance"]

            # Diagnostic logging at DEBUG level
            logger.debug(
                f"HDRF '{hdrf_id}': L dims={L_actual.dims}, E dims={E_reference.dims}"
            )
            logger.debug(
                f"  L_actual: mean={float(L_actual.mean()):.6e}, "
                f"range=[{float(L_actual.min()):.6e}, {float(L_actual.max()):.6e}]"
            )
            logger.debug(
                f"  E_reference: mean={float(E_reference.mean()):.6e}, "
                f"range=[{float(E_reference.min()):.6e}, {float(E_reference.max()):.6e}]"
            )

            # Compute HDRF = (π × L_actual) / E_reference
            hdrf = (np.pi * L_actual) / E_reference

            logger.info(
                f"  HDRF '{hdrf_id}': mean={float(hdrf.mean()):.4f}, "
                f"range=[{float(hdrf.min()):.4f}, {float(hdrf.max()):.4f}]"
            )

            # Create dataset
            hdrf_dataset = xr.Dataset(
                {
                    "hdrf": hdrf,
                    "radiance_actual": L_actual,
                    "irradiance_reference": E_reference,
                },
                coords=hdrf.coords,
            )

            hdrf_dataset.attrs.update(
                {
                    "quantity": "hdrf",
                    "units": "dimensionless",
                    "method": "radiance_irradiance_ratio",
                    "reference_irradiance_id": ref_id,
                }
            )

            hdrf_datasets[hdrf_id] = hdrf_dataset

            # Save
            output_file = output_dir / f"{hdrf_id}_hdrf.zarr"
            hdrf_dataset.to_zarr(output_file, mode="w")
            logger.debug(f"Saved HDRF to {output_file.name}")

        logger.info(f"HDRF computation complete: {len(hdrf_datasets)} measurements")

        return hdrf_datasets
