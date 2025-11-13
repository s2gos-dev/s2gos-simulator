"""Terrain elevation querying utilities for scene simulations.

This module provides a unified interface for querying terrain elevation from
DEM (Digital Elevation Model) data, eliminating code duplication across
different processors and backends.
"""

import logging
from typing import Optional, Tuple

import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from s2gos_utils.io.paths import UPath
from s2gos_utils.scene.description import SceneDescription

logger = logging.getLogger(__name__)


class TerrainQuery:
    """Unified terrain elevation querying for simulation backends.

    This class provides methods to:
    - Find DEM files in scene directories
    - Query elevation at geographic coordinates (lat/lon)
    - Query elevation at scene coordinates (x, y)
    - Validate coordinate bounds
    """

    def __init__(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
    ):
        """Initialize terrain query for a specific scene.

        Args:
            scene_description: Scene description containing metadata
            scene_dir: Scene directory containing DEM data
        """
        self.scene_description = scene_description
        self.scene_dir = scene_dir
        self._dem_path: Optional[UPath] = None
        self._dem_data: Optional[xr.DataArray] = None

    def find_dem_file(self) -> Optional[UPath]:
        """Find DEM file in scene data directory.

        Searches for DEM files in the following order:
        1. dem_{scene_name}_{resolution}m.zarr
        2. dem_{scene_name}.zarr

        Returns:
            Path to DEM file or None if not found
        """
        if self._dem_path is not None:
            return self._dem_path

        scene_name = self.scene_description.name
        resolution = self.scene_description.resolution_m

        # Try resolution-specific DEM first
        dem_path = self.scene_dir / "data" / f"dem_{scene_name}_{resolution}m.zarr"
        if dem_path.exists():
            logger.debug(f"Found DEM at: {dem_path}")
            self._dem_path = dem_path
            return dem_path

        # Fall back to generic DEM
        dem_path = self.scene_dir / "data" / f"dem_{scene_name}.zarr"
        if dem_path.exists():
            logger.debug(f"Found DEM at: {dem_path}")
            self._dem_path = dem_path
            return dem_path

        logger.warning(
            f"DEM not found in {self.scene_dir / 'data'}. "
            f"Tried: dem_{scene_name}_{resolution}m.zarr, dem_{scene_name}.zarr"
        )
        return None

    def query_elevation_at_scene_coords(
        self,
        x: float,
        y: float,
        raise_on_error: bool = True,
    ) -> float:
        """Query terrain elevation at scene coordinates (x, y) in meters.

        Args:
            x: Scene x-coordinate in meters
            y: Scene y-coordinate in meters
            raise_on_error: If True, raise exceptions. If False, return 0.0 on error.

        Returns:
            Elevation in meters at (x, y)

        Raises:
            FileNotFoundError: If DEM file not found (only if raise_on_error=True)
        """
        dem_path = self.find_dem_file()

        if dem_path is None:
            msg = (
                f"DEM file not found in {self.scene_dir / 'data'}. "
                f"Cannot query terrain elevation."
            )
            if raise_on_error:
                raise FileNotFoundError(msg)
            else:
                logger.warning(msg + " Using 0.0m.")
                return 0.0

        try:
            # Load DEM data
            with xr.open_zarr(dem_path) as dem_ds:
                dem_data = dem_ds["elevation"]

                # Create interpolator
                interpolator = RegularGridInterpolator(
                    (dem_data.y.values, dem_data.x.values),
                    dem_data.values,
                    method="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )

                # Query elevation
                elevation = float(interpolator([(y, x)])[0])

            logger.debug(
                f"DEM elevation at scene coords ({x:.1f}m, {y:.1f}m): {elevation:.2f}m"
            )
            return elevation

        except Exception as e:
            msg = f"DEM query failed at ({x}, {y}): {e}"
            if raise_on_error:
                raise RuntimeError(msg) from e
            else:
                logger.warning(msg + " Using 0.0m.")
                return 0.0

    def query_elevation_at_geographic_coords(
        self,
        lat: float,
        lon: float,
        raise_on_error: bool = False,
    ) -> float:
        """Query terrain elevation at geographic coordinates (lat, lon).

        This method handles coordinate transformation from lat/lon to scene
        coordinates before querying the DEM.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            raise_on_error: If True, raise exceptions. If False, return 0.0 on error.

        Returns:
            Elevation in meters at (lat, lon)
        """
        # Transform lat/lon to scene coordinates
        try:
            x, y = self.latlon_to_scene(lat, lon)
            logger.debug(
                f"Geographic coords ({lat:.6f}°, {lon:.6f}°) -> "
                f"scene coords ({x:.1f}m, {y:.1f}m)"
            )
        except Exception as e:
            msg = f"Coordinate transformation failed for ({lat}, {lon}): {e}"
            if raise_on_error:
                raise RuntimeError(msg) from e
            else:
                logger.warning(msg + " Using 0.0m elevation.")
                return 0.0

        # Query elevation at scene coordinates
        return self.query_elevation_at_scene_coords(x, y, raise_on_error=raise_on_error)

    def latlon_to_scene(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert geographic coordinates to scene coordinates.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Tuple of (x, y) scene coordinates in meters

        Raises:
            ValueError: If coordinate transformation fails
        """
        from s2gos_utils.coordinates import CoordinateSystem

        center_lat = self.scene_description.location["center_lat"]
        center_lon = self.scene_description.location["center_lon"]

        coord_system = CoordinateSystem(center_lat, center_lon)
        x, y = coord_system.latlon_to_scene(lat, lon)

        return x, y

    def validate_coordinate_bounds(
        self,
        lat: float,
        lon: float,
        max_distance_km: float = 50.0,
    ) -> bool:
        """Validate that coordinates are within acceptable bounds.

        The local tangent plane approximation used for coordinate transformation
        breaks down at large distances (typically >50km from scene center).

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            max_distance_km: Maximum allowed distance from scene center in km

        Returns:
            True if coordinates are valid, False otherwise
        """
        try:
            x, y = self.latlon_to_scene(lat, lon)
            distance_m = (x**2 + y**2) ** 0.5
            distance_km = distance_m / 1000.0

            if distance_km > max_distance_km:
                logger.warning(
                    f"Coordinate ({lat}, {lon}) is {distance_km:.1f}km from scene center. "
                    f"Local tangent plane approximation may be inaccurate (>{max_distance_km}km limit)."
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Coordinate validation failed: {e}")
            return False

    def get_dem_bounds(self) -> Optional[dict]:
        """Get DEM spatial bounds in scene coordinates.

        Returns:
            Dictionary with 'x_min', 'x_max', 'y_min', 'y_max' or None if DEM not found
        """
        dem_path = self.find_dem_file()
        if dem_path is None:
            return None

        try:
            with xr.open_zarr(dem_path) as dem_ds:
                dem_data = dem_ds["elevation"]
                return {
                    "x_min": float(dem_data.x.min()),
                    "x_max": float(dem_data.x.max()),
                    "y_min": float(dem_data.y.min()),
                    "y_max": float(dem_data.y.max()),
                }
        except Exception as e:
            logger.error(f"Failed to get DEM bounds: {e}")
            return None

    def validate_scene_coordinates(
        self,
        x: float,
        y: float,
    ) -> bool:
        """Validate that scene coordinates are within DEM bounds.

        Args:
            x: Scene x-coordinate in meters
            y: Scene y-coordinate in meters

        Returns:
            True if coordinates are within DEM bounds, False otherwise
        """
        bounds = self.get_dem_bounds()
        if bounds is None:
            logger.warning("Cannot validate coordinates - DEM not found")
            return False

        if not (bounds["x_min"] <= x <= bounds["x_max"]):
            logger.warning(
                f"X coordinate {x}m outside DEM bounds "
                f"[{bounds['x_min']:.1f}, {bounds['x_max']:.1f}]m"
            )
            return False

        if not (bounds["y_min"] <= y <= bounds["y_max"]):
            logger.warning(
                f"Y coordinate {y}m outside DEM bounds "
                f"[{bounds['y_min']:.1f}, {bounds['y_max']:.1f}]m"
            )
            return False

        return True
