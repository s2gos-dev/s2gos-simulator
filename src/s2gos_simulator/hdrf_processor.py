import logging
from datetime import datetime
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
        from .config import MeasurementType

        for sensor in self.simulation_config.sensors:
            if hasattr(sensor, "produces"):
                if MeasurementType.HDRF in sensor.produces:
                    return True

        for rq in self.simulation_config.radiative_quantities:
            if rq.quantity == MeasurementType.HDRF:
                return True

        return False

    def get_hdrf_measure_ids(self) -> List[str]:
        """Get list of measure IDs that produce HDRF.

        Returns:
            List of measure IDs that require HDRF computation
        """
        from .config import MeasurementType

        hdrf_ids = []

        for sensor in self.simulation_config.sensors:
            if hasattr(sensor, "produces"):
                if MeasurementType.HDRF in sensor.produces:
                    hdrf_ids.append(sensor.id)

        for rq in self.simulation_config.radiative_quantities:
            if rq.quantity == MeasurementType.HDRF:
                if rq.id:
                    rq_id = rq.id
                else:
                    string_viewing_zenith = str(rq.viewing_zenith).replace(".", "_")
                    string_viewing_azimuth = str(rq.viewing_azimuth).replace(".", "_")
                    rq_id = f"hdrf_{string_viewing_zenith}__{string_viewing_azimuth}"
                hdrf_ids.append(rq_id)

        return [x.replace(".", "_") for x in hdrf_ids]

    def _get_hdrf_config_for_measure(
        self, measure_id: str
    ) -> Optional[Tuple[float, float, float]]:
        """Get target location and height offset for an HDRF measure.

        Args:
            measure_id: HDRF measure ID

        Returns:
            Tuple of (target_lat, target_lon, height_offset_m) or None if not found
        """
        from .config import MeasurementType

        for sensor in self.simulation_config.sensors:
            if hasattr(sensor, "produces") and MeasurementType.HDRF in sensor.produces:
                if sensor.id == measure_id or sensor.id.replace(".", "_") == measure_id:
                    if hasattr(sensor, "target_center_lat") and hasattr(
                        sensor, "reference_panel_offset_m"
                    ):
                        return (
                            sensor.target_center_lat,
                            sensor.target_center_lon,
                            sensor.reference_panel_offset_m,
                        )

        for rq in self.simulation_config.radiative_quantities:
            if rq.quantity == MeasurementType.HDRF:
                if rq.id:
                    rq_id = rq.id
                else:
                    string_viewing_zenith = str(rq.viewing_zenith).replace(".", "_")
                    string_viewing_azimuth = str(rq.viewing_azimuth).replace(".", "_")
                    rq_id = f"hdrf_{string_viewing_zenith}__{string_viewing_azimuth}"

                if rq_id == measure_id:
                    return (
                        rq.target_lat,  # Will use scene center
                        rq.target_lon,
                        rq.reference_panel_offset_m,
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
            scene_description, scene_dir, lat, lon, height_offset,
            disk_id="white_reference_disk_hdrf"
        )

    def execute_dual_simulation(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: UPath,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """Execute dual simulation: actual scene + white reference scene."""
        logger.info("=" * 60)
        logger.info("HDRF Dual Simulation")
        logger.info("=" * 60)

        # Create white reference scene
        reference_scene, self.reference_panel_coords = (
            self.create_white_reference_scene(scene_description, scene_dir)
        )
        self.backend.reference_panel_coords = self.reference_panel_coords

        # Run both simulations
        logger.info("\nActual scene simulation...")
        actual_results = self._run_simulation(
            scene_description, scene_dir, output_dir / "actual", run_all_measures=True
        )

        logger.info("\nWhite reference simulation...")
        reference_results = self._run_simulation(
            reference_scene, scene_dir, output_dir / "reference", run_all_measures=False
        )

        logger.info(f"\n{'=' * 60}\n")
        return actual_results, reference_results

    def _run_simulation(
        self,
        scene: SceneDescription,
        scene_dir: UPath,
        output_dir: UPath,
        run_all_measures: bool,
    ) -> xr.Dataset:
        """Run simulation, optionally filtering to HDRF measures only."""
        import eradiate
        from s2gos_utils.io.paths import mkdir

        mkdir(output_dir)

        experiment = self.backend._create_experiment(scene, scene_dir)

        if run_all_measures:
            # Run all configured measures
            for i in range(len(experiment.measures)):
                eradiate.run(experiment, measures=i)
        else:
            # Run only HDRF measures (skip others on white reference scene)
            hdrf_ids = set(self.get_hdrf_measure_ids())
            for i, measure in enumerate(experiment.measures):
                measure_id = getattr(measure, "id", f"measure_{i}")
                if measure_id in hdrf_ids:
                    eradiate.run(experiment, measures=i)

        return self.backend._process_results(experiment, output_dir)

    def compute_hdrf(
        self,
        actual_results: Dict[str, xr.Dataset],
        reference_results: Dict[str, xr.Dataset],
    ) -> Dict[str, xr.Dataset]:
        """Compute HDRF = L_actual / L_reference for each measure."""
        logger.info("\nComputing HDRF...")

        hdrf_datasets = {}
        for measure_id in self.get_hdrf_measure_ids():
            if measure_id not in actual_results or measure_id not in reference_results:
                logger.warning(f"Skipping {measure_id} (missing results)")
                continue

            actual_ds = actual_results[measure_id]
            reference_ds = reference_results[measure_id]

            if "radiance" not in actual_ds or "radiance" not in reference_ds:
                logger.warning(f"Skipping {measure_id} (no radiance)")
                continue

            L_actual = actual_ds["radiance"].squeeze(drop=True)
            L_ref = reference_ds["radiance"].squeeze(drop=True)

            if L_actual.dims != L_ref.dims:
                logger.error(f"Skipping {measure_id} (dim mismatch)")
                continue

            # HDRF = L_actual / L_reference (average reference to reduce noise)
            hdrf = L_actual / L_ref.mean()

            logger.info(
                f"  {measure_id}: mean={float(hdrf.mean()):.3f}, "
                f"range=[{float(hdrf.min()):.3f}, {float(hdrf.max()):.3f}]"
            )

            hdrf_dataset = xr.Dataset(
                {"hdrf": hdrf, "radiance_actual": L_actual, "radiance_reference": L_ref},
                coords=hdrf.coords,
            )

            hdrf_dataset.attrs.update({
                "quantity": "hdrf",
                "units": "dimensionless",
                "method": "dual_simulation",
                "timestamp": datetime.now().isoformat(),
            })

            hdrf_datasets[measure_id] = hdrf_dataset

        logger.info(f"Complete: {len(hdrf_datasets)} HDRF datasets\n")
        return hdrf_datasets

    def save_hdrf_results(
        self, hdrf_datasets: Dict[str, xr.Dataset], output_dir: UPath
    ) -> None:
        """Save HDRF results to disk."""
        from s2gos_utils.io.paths import mkdir

        hdrf_dir = output_dir / "hdrf_results"
        mkdir(hdrf_dir)

        for measure_id, dataset in hdrf_datasets.items():
            output_file = hdrf_dir / f"{measure_id}_hdrf.zarr"
            dataset.to_zarr(output_file, mode="w")
            logger.info(f"  Saved: {output_file.name}")
