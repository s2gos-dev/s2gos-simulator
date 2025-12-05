"""BOA irradiance measurement using reference disk technique.

Measures downward irradiance at BOA by:
1. Placing a small reference disk with Lambertian reflectance (ρ=1.0) at target location
2. Running simulation with hemispherical distant sensor
3. Converting measured radiance to irradiance: E = π × L_mean
"""

import logging
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from s2gos_utils.scene import SceneDescription
from upath import UPath

logger = logging.getLogger(__name__)

REFERENCE_DISK_RADIUS_M = 0.01  # 1cm radius for consistent BOA measurement


class IrradianceProcessor:
    """Processor for BOA irradiance measurements using reference disk technique."""

    def __init__(self, backend):
        self.backend = backend
        self.simulation_config = backend.simulation_config

    def requires_irradiance(self) -> bool:
        """Check if any irradiance measurements are configured."""
        from .config import IrradianceConfig

        return any(
            isinstance(m, IrradianceConfig) for m in self.simulation_config.measurements
        )

    def _to_scene_coords(
        self, lat: float, lon: float, scene: SceneDescription
    ) -> Tuple[float, float]:
        """Convert lat/lon to scene XY coordinates."""
        from s2gos_utils.coordinates import CoordinateSystem

        coord_sys = CoordinateSystem(
            scene.location["center_lat"], scene.location["center_lon"]
        )
        return coord_sys.latlon_to_scene(lat, lon)

    def _to_lat_lon_coords(
        self, x: float, y: float, scene: SceneDescription
    ) -> Tuple[float, float]:
        """Convert lat/lon to scene XY coordinates."""
        from s2gos_utils.coordinates import CoordinateSystem

        coord_sys = CoordinateSystem(
            scene.location["center_lat"], scene.location["center_lon"]
        )
        return coord_sys.scene_to_latlon(x, y)

    def _get_disk_elevation(
        self,
        scene: SceneDescription,
        scene_dir: UPath,
        lat: float,
        lon: float,
        height_offset_m: float,
    ) -> float:
        """Get disk elevation: terrain + height_offset_m."""
        from s2gos_simulator.terrain_query import TerrainQuery

        terrain_elev = TerrainQuery(
            scene, scene_dir
        ).query_elevation_at_geographic_coords(lat, lon, raise_on_error=False)
        disk_elev = terrain_elev + height_offset_m

        logger.info(
            f"Disk at ({lat:.4f}, {lon:.4f}): "
            f"terrain={terrain_elev:.1f}m + offset={height_offset_m:.1f}m = {disk_elev:.1f}m"
        )
        return disk_elev

    def create_reference_disk_scene(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        location,
        height_offset_m: float,
        disk_id: str = "boa_irradiance_disk",
    ) -> Tuple[SceneDescription, Tuple[float, float, float]]:
        """Create scene with white Lambertian disk (ρ=1.0) at target location.

        Returns modified scene and disk coordinates (x, y, z).
        """
        from dataclasses import replace

        from s2gos_simulator.terrain_query import TerrainQuery

        if height_offset_m < 0:
            raise ValueError(f"height_offset_m must be >= 0, got {height_offset_m}")

        target_lat, target_lon = location.target_lat, location.target_lon

        tq = TerrainQuery(scene_description, scene_dir)
        if not tq.validate_coordinate_bounds(
            target_lat, target_lon, max_distance_km=50.0
        ):
            logger.warning(
                f"Coordinates ({target_lat}, {target_lon}) far from scene center"
            )

        # Get disk position
        if location.target_x is None:
            x, y = self._to_scene_coords(target_lat, target_lon, scene_description)
            z = self._get_disk_elevation(
                scene_description, scene_dir, target_lat, target_lon, height_offset_m
            )
        else:
            x, y, z = location.target_x, location.target_y, location.target_z
            target_lat, target_lon = self._to_lat_lon_coords(x, y, scene_description)

            if location.terrain_relative_height:
                z = self._get_disk_elevation(
                    scene_description,
                    scene_dir,
                    target_lat,
                    target_lon,
                    location.height_offset_m,
                )

        # Create disk object
        reference_disk = {
            "object_id": disk_id,
            "type": "disk",
            "center": [x, y, z],
            "radius": REFERENCE_DISK_RADIUS_M,
        }

        # Shallow copy scene with new objects list (avoid modifying original)
        new_objects = (scene_description.objects or []).copy()
        new_objects.insert(0, reference_disk)
        disk_scene = replace(scene_description, objects=new_objects)

        logger.info(f"Created reference disk at ({x:.1f}, {y:.1f}, {z:.1f})")
        return disk_scene, (x, y, z)

    def convert_radiance_to_irradiance(
        self,
        radiance: xr.DataArray,
        measurement_config: "IrradianceConfig" = None,
    ) -> xr.DataArray:
        """Convert disk radiance to BOA irradiance: E = π × L_mean.

        Averages over hemisphere sampling dimensions (from hdistant measure),
        preserves wavelength dimension.
        """
        # Average over all non-wavelength dims (hemisphere sampling from hdistant measure)
        wavelength_dims = {"w", "wavelength", "lambda", "wl"}
        hemisphere_dims = [d for d in radiance.dims if d not in wavelength_dims]

        L_mean = radiance.mean(dim=hemisphere_dims) if hemisphere_dims else radiance
        E_boa = np.pi * L_mean  # E = π × L for Lambertian ρ=1.0

        logger.info(f"BOA irradiance: mean={float(E_boa.mean()):.3e} W/m²/nm")

        # Minimal metadata
        E_boa.attrs.update(
            {
                "quantity": "boa_irradiance",
                "units": "W m^-2 nm^-1",
                "conversion": "E = π × mean(L)",
            }
        )

        # if measurement_config:
        #     E_boa.attrs.update(
        #         {
        #             "lat": measurement_config.target_lat,
        #             "lon": measurement_config.target_lon,
        #             "height_offset_m": measurement_config.height_offset_m,
        #         }
        #     )

        return E_boa

    def execute_irradiance_measurements(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: UPath,
    ) -> Dict[str, xr.Dataset]:
        """Execute BOA irradiance measurements: disk scene → simulation → E = π × L."""
        import eradiate
        from s2gos_utils.io.paths import mkdir

        logger.info("=" * 60)
        logger.info("BOA Irradiance Measurements")
        logger.info("=" * 60)
        mkdir(output_dir)

        # Store disk coords for backend (TODO: refactor to avoid this side effect)
        if not hasattr(self.backend, "irradiance_disk_coords"):
            self.backend.irradiance_disk_coords = {}

        results = {}
        # Filter for IrradianceConfig instances from unified measurements list
        from .config import IrradianceConfig

        irradiance_configs = [
            m
            for m in self.simulation_config.measurements
            if isinstance(m, IrradianceConfig)
        ]

        for config in irradiance_configs:
            logger.info(f"\n[{config.id}]")
            print(config)
            # Create disk scene
            disk_scene, disk_coords = self.create_reference_disk_scene(
                scene_description,
                scene_dir,
                config.location,
                config.location.height_offset_m,
                disk_id=f"disk_{config.id}",
            )
            self.backend.irradiance_disk_coords[config.id] = disk_coords

            # Run simulation
            experiment = self.backend._create_experiment(disk_scene, scene_dir)

            # Build measure ID → index mapping for efficient lookup
            measure_map = {
                getattr(m, "id", f"measure_{i}"): i
                for i, m in enumerate(experiment.measures)
            }

            if config.id not in measure_map:
                raise RuntimeError(
                    f"Measure '{config.id}' not found. Available: {list(measure_map.keys())}"
                )

            measure_idx = measure_map[config.id]
            eradiate.run(experiment, measures=measure_idx)

            # Extract results (keyed by measure ID string, not index)
            result = experiment.results[config.id]
            if "radiance" not in result:
                raise RuntimeError(
                    f"No 'radiance' in results. Available: {list(result.data_vars)}"
                )

            # Convert to irradiance
            E_boa = self.convert_radiance_to_irradiance(result["radiance"], config)

            # Include TOA irradiance if available
            dataset_vars = {"boa_irradiance": E_boa}
            if "irradiance" in result:
                wavelength_dims = {"w", "wavelength", "lambda", "wl"}
                E_toa = result["irradiance"]
                toa_dims = [d for d in E_toa.dims if d not in wavelength_dims]
                if toa_dims:
                    E_toa = E_toa.mean(dim=toa_dims)
                E_toa.attrs.update(
                    {"quantity": "toa_irradiance", "units": "W m^-2 nm^-1"}
                )
                dataset_vars["toa_irradiance"] = E_toa

            result_ds = xr.Dataset(dataset_vars)

            # Save
            output_file = output_dir / f"{config.id}.zarr"
            result_ds.to_zarr(output_file, mode="w")
            logger.info(f"  ✓ Saved {output_file.name}")

            results[config.id] = result_ds

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Complete: {len(results)} measurements")
        logger.info(f"{'=' * 60}\n")
        return results
