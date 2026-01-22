from typing import Optional

import numpy as np
import xarray as xr


class SensorProcessor:
    """Unified sensor processor for all sensor types.

    Currently supported:
    - HYPSTAR: Gaussian SRF convolution + spatial averaging
    """

    def __init__(self, simulation_config):
        """Initialize sensor processor.

        Args:
            simulation_config: SimulationConfig containing sensor definitions
        """
        self.simulation_config = simulation_config

    def process_sensor_result(self, dataset: xr.Dataset, sensor) -> xr.Dataset:
        """Apply sensor-specific post-processing to a single dataset.

        This is the main entry point for all sensor post-processing.
        Dispatches to instrument-specific handlers based on sensor type.

        Args:
            dataset: Raw simulation result dataset
            sensor: Sensor configuration object

        Returns:
            Post-processed dataset with SRF convolution, averaging, etc.
        """
        return self._apply_post_processing(dataset, sensor)

    def _find_sensor_for_result(self, result_id: str):
        """Find sensor associated with a result ID.

        Checks both measurements (which reference sensors by ID) and
        direct sensor outputs.

        Args:
            result_id: Result identifier (measurement ID or sensor ID)

        Returns:
            Sensor object or None if not found
        """
        for measurement in self.simulation_config.measurements:
            if getattr(measurement, "id", None) == result_id:
                sensor_id = getattr(measurement, "sensor_id", None)
                if sensor_id:
                    return self._find_sensor_by_id(sensor_id)

        return self._find_sensor_by_id(result_id)

    def _find_sensor_by_id(self, sensor_id: str):
        """Find sensor by ID.

        Args:
            sensor_id: Sensor identifier

        Returns:
            Sensor object or None if not found
        """
        for sensor in self.simulation_config.sensors:
            if sensor.id == sensor_id:
                return sensor
        return None

    def _apply_post_processing(self, dataset: xr.Dataset, sensor) -> xr.Dataset:
        """Apply sensor-specific post-processing.

        Dispatches to the appropriate processing method based on sensor type.
        Processing order:
        1. Check for Gaussian SRF (works for any sensor with gaussian SRF type)
        2. Fall back to instrument-specific processing (HYPSTAR, etc.)

        Args:
            dataset: Raw simulation result
            sensor: Sensor that produced the result

        Returns:
            Post-processed dataset
        """
        import logging

        from ..config import GroundInstrumentType, GroundSensor, SpectralResponse

        logger = logging.getLogger(__name__)

        # Check for Gaussian SRF type - applies to any sensor (satellite, ground, UAV)
        if hasattr(sensor, "srf") and sensor.srf is not None:
            if (
                isinstance(sensor.srf, SpectralResponse)
                and sensor.srf.type == "gaussian"
            ):
                logger.info(
                    f"Applying Gaussian SRF post-processing to sensor '{sensor.id}'"
                )
                return self._apply_gaussian_srf_from_config(dataset, sensor.srf)

        # Instrument-specific processing (HYPSTAR has its own SRF handling)
        if isinstance(sensor, GroundSensor):
            if sensor.instrument == GroundInstrumentType.HYPSTAR:
                return self._apply_hypstar_post_processing(dataset, sensor)

        return dataset

    def _apply_gaussian_srf_from_config(
        self, dataset: xr.Dataset, srf_config
    ) -> xr.Dataset:
        """Apply Gaussian SRF convolution using SpectralResponse configuration.

        This is a generalized method that works for any sensor with gaussian SRF type,
        including CHIME satellite (2D imagery), HYPSTAR (point measurements after
        spatial averaging), and future hyperspectral instruments.

        The Gaussian convolution is applied to radiance data at each target wavelength,
        using wavelength-dependent FWHM from spectral_regions or a single fwhm_nm.

        For 2D satellite data (with x_index, y_index dimensions), the spatial structure
        is preserved and the output has shape (n_wavelengths, ny, nx).

        Args:
            dataset: Raw simulation result dataset with radiance data
            srf_config: SpectralResponse with type="gaussian"

        Returns:
            Dataset with Gaussian SRF convolution applied to radiance.
            Preserves spatial dimensions for 2D data (satellite imagery).
        """
        import logging

        import eradiate
        from eradiate.pipelines.logic import apply_spectral_response
        from eradiate.spectral.response import make_gaussian

        logger = logging.getLogger(__name__)

        if "radiance" not in dataset:
            logger.warning("No radiance data found, skipping Gaussian SRF processing")
            return dataset

        radiance = dataset["radiance"]
        print(dataset)
        # Generate target wavelengths from output_grid
        target_wavelengths = srf_config.output_grid.generate_wavelengths()
        logger.info(
            f"Applying Gaussian SRF to {len(target_wavelengths)} wavelengths "
            f"({target_wavelengths.min():.1f}-{target_wavelengths.max():.1f} nm)"
        )

        # Check required bin edge coordinates for SRF application
        required_coords = {"bin_wmin", "bin_wmax"}
        if not required_coords.issubset(set(radiance.coords.keys())):
            raise ValueError(
                f"Radiance missing required SRF coordinates: {required_coords}. "
                f"Available: {set(radiance.coords.keys())}. "
                f"Eradiate simulations should include bin edge coordinates."
            )

        # Detect if this is 2D satellite data (has spatial dimensions)
        spatial_dims = [d for d in ["x_index", "y_index"] if d in radiance.dims]
        is_2d_data = len(spatial_dims) == 2

        if is_2d_data:
            logger.info(
                f"Processing 2D satellite data with spatial dims: {spatial_dims}"
            )

        # Apply Gaussian SRF at each target wavelength
        processed_bands = []
        for wl in target_wavelengths:
            fwhm = srf_config.get_fwhm_for_wavelength(wl)
            gaussian_srf = make_gaussian(wl, fwhm, pad=True)
            band_srf = eradiate.spectral.BandSRF.from_dataarray(gaussian_srf.srf)
            l_srf = apply_spectral_response(radiance, band_srf)
            processed_bands.append(l_srf)

        # Build result DataArray - preserve spatial dimensions for 2D data
        if is_2d_data:
            # Stack along wavelength dimension, preserving spatial dims
            # Each l_srf has shape (ny, nx) or (ny, nx, saa, sza, ...), stack along new 'w' dim
            result_radiance = xr.concat(processed_bands, dim="w")
            result_radiance = result_radiance.assign_coords(w=target_wavelengths)
            # Reorder dims to put 'w' first, then spatial dims, then any others
            # Use ellipsis (...) to handle any additional dimensions (saa, sza, etc.)
            other_dims = [
                d for d in result_radiance.dims if d not in ["w", "y_index", "x_index"]
            ]
            result_radiance = result_radiance.transpose(
                "w", "y_index", "x_index", *other_dims
            )
        else:
            # 1D case (point measurement or already spatially averaged)
            data_array = np.array(
                [float(band.values.squeeze()) for band in processed_bands],
                dtype=np.float64,
            )
            result_radiance = xr.DataArray(
                data=data_array,
                dims=["w"],
                coords={"w": target_wavelengths},
            )

        result_radiance = result_radiance.sortby("w")
        result_radiance.attrs = {
            **radiance.attrs,
            "gaussian_srf_applied": True,
            "srf_type": "gaussian",
        }
        toa_irradiance_interpolated = dataset.irradiance.interp(w=result_radiance.w)
        return xr.Dataset(
            {
                "radiance": result_radiance,
                "toa_irradiance": toa_irradiance_interpolated,
            },
            attrs={**dataset.attrs, "gaussian_srf_post_processed": True},
        )

    def _apply_hypstar_post_processing(self, dataset: xr.Dataset, sensor) -> xr.Dataset:
        """Apply HYPSTAR-specific post-processing.

        Pipeline order:
        1. Circular FOV masking (default: enabled, sets pixels outside FOV to NaN)
        2. RGB image generation (optional, requires spatial dims)
        3. Spatial averaging (nanmean over x_index, y_index)
        4. Gaussian SRF convolution (wavelength processing)

        Args:
            dataset: Raw simulation result with radiance data
            sensor: GroundSensor with HYPSTAR instrument

        Returns:
            Post-processed dataset
        """
        import logging

        from upath import UPath

        logger = logging.getLogger(__name__)

        config = sensor.hypstar_post_processing
        if config is None:
            return dataset

        result = dataset.copy()
        radiance = result["radiance"]

        if getattr(config, "apply_circular_mask", True):
            fov = getattr(sensor, "fov", None)
            if fov is not None:
                radiance = self._apply_circular_fov_mask(radiance, fov)
            else:
                logger.warning(
                    "Circular mask enabled but sensor has no FOV parameter. Skipping."
                )

        if getattr(config, "generate_rgb_image", False):
            output_dir = UPath(result.attrs.get("output_dir", "./output"))
            self._generate_rgb_image(radiance, sensor.id, output_dir, config)

        if getattr(config, "spatial_averaging", True):
            radiance = self._apply_spatial_averaging(radiance)

        if getattr(config, "apply_srf", True):
            fwhm_vnir = getattr(config, "fwhm_vnir_nm", 3.0)
            fwhm_swir = getattr(config, "fwhm_swir_nm", 10.0)

            target_wavelengths = None
            if config.real_reference_file is not None:
                try:
                    logger.info(
                        f"Loading HYPSTAR reference: {config.real_reference_file}"
                    )
                    real_reference_ds = xr.open_dataset(config.real_reference_file)
                    target_wavelengths = real_reference_ds[config.wavelength_variable]
                    logger.info(
                        f"Loaded {len(target_wavelengths)} wavelengths "
                        f"({float(target_wavelengths.min()):.1f}-{float(target_wavelengths.max()):.1f} nm)"
                    )
                except Exception as e:
                    logger.error(f"Failed to load HYPSTAR reference: {e}")
                    target_wavelengths = None

            radiance = self._apply_gaussian_srf(
                radiance,
                fwhm_vnir=fwhm_vnir,
                fwhm_swir=fwhm_swir,
                target_wavelengths=target_wavelengths,
            )

        result = xr.Dataset(
            {"radiance": radiance},
            attrs={**result.attrs, "hypstar_post_processed": True},
        )
        return result

    def _apply_spatial_averaging(self, radiance: xr.DataArray) -> xr.DataArray:
        """Average over spatial dimensions (FOV pixels).

        Uses nanmean to properly handle NaN values from circular masking.
        Pixels outside the circular FOV are excluded from averaging.

        Args:
            radiance: Radiance DataArray with spatial dimensions

        Returns:
            Radiance averaged over spatial dimensions
        """
        import numpy as np

        spatial_dims = [d for d in ["x_index", "y_index"] if d in radiance.dims]
        if spatial_dims:
            return radiance.reduce(np.nanmean, dim=spatial_dims)
        return radiance

    def _apply_circular_fov_mask(
        self,
        radiance: xr.DataArray,
        fov_degrees: float,
    ) -> xr.DataArray:
        """Apply circular FOV mask to camera data.

        Sets pixels outside the circular field of view to NaN.
        Uses simple geometric approach: circle radius = min(width, height) / 2.

        Args:
            radiance: Radiance DataArray with spatial dimensions (x_index, y_index)
            fov_degrees: Field of view in degrees (for metadata only)

        Returns:
            Radiance with circular mask applied (values outside circle are NaN)
        """
        import logging

        import numpy as np

        logger = logging.getLogger(__name__)

        if "x_index" not in radiance.dims or "y_index" not in radiance.dims:
            logger.warning(
                "Circular mask requires x_index and y_index dimensions. Skipping."
            )
            return radiance

        ny, nx = len(radiance.y_index), len(radiance.x_index)

        center_x = (nx - 1) / 2.0
        center_y = (ny - 1) / 2.0
        radius_pixels = min(nx, ny) / 2.0

        y_coords, x_coords = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        circular_mask_array = distances <= radius_pixels

        circular_mask = xr.DataArray(
            circular_mask_array,
            dims=["y_index", "x_index"],
            coords={
                "y_index": radiance.coords["y_index"],
                "x_index": radiance.coords["x_index"],
            },
        )

        masked_radiance = radiance.where(circular_mask, other=np.nan)

        masked_radiance.attrs["circular_mask_applied"] = True
        masked_radiance.attrs["fov_degrees"] = fov_degrees
        masked_radiance.attrs["mask_radius_pixels"] = float(radius_pixels)

        logger.info(
            f"Applied circular FOV mask: radius={radius_pixels:.1f} pixels, "
            f"FOV={fov_degrees}°"
        )

        return masked_radiance

    def _generate_rgb_image(
        self,
        radiance: xr.DataArray,
        sensor_id: str,
        output_dir,
        config,
    ) -> None:
        """Generate RGB visualization from spatial radiance data.

        Creates RGB image by selecting wavelengths closest to red, green, blue
        target wavelengths and saving as PNG.

        Args:
            radiance: Radiance DataArray with spatial and spectral dimensions
            sensor_id: Sensor identifier for output filename
            output_dir: Directory to save RGB image
            config: Post-processing configuration with RGB settings
        """
        import logging

        import numpy as np
        from PIL import Image
        from s2gos_utils.io.paths import mkdir, open_file

        logger = logging.getLogger(__name__)

        if "x_index" not in radiance.dims or "y_index" not in radiance.dims:
            logger.warning("RGB generation requires spatial dimensions. Skipping.")
            return

        if "w" not in radiance.dims:
            logger.warning("RGB generation requires wavelength dimension. Skipping.")
            return

        wavelengths = radiance.coords["w"].values
        if len(wavelengths) < 3:
            logger.warning(
                f"RGB generation requires at least 3 wavelengths "
                f"(found {len(wavelengths)}). Skipping."
            )
            return

        try:
            from eradiate.xarray.interp import dataarray_to_rgb
        except ImportError:
            logger.warning(
                "eradiate.xarray.interp.dataarray_to_rgb not available. Skipping RGB."
            )
            return

        target_r, target_g, target_b = config.rgb_wavelengths
        actual_wavelengths = [
            radiance.sel(w=target_r, method="nearest").w.item(),
            radiance.sel(w=target_g, method="nearest").w.item(),
            radiance.sel(w=target_b, method="nearest").w.item(),
        ]

        logger.info(
            f"RGB generation for {sensor_id}: "
            f"R={actual_wavelengths[0]:.1f}nm, "
            f"G={actual_wavelengths[1]:.1f}nm, "
            f"B={actual_wavelengths[2]:.1f}nm"
        )

        channels = [("w", w) for w in actual_wavelengths]
        img = dataarray_to_rgb(radiance, channels=channels, normalize=False)

        img = img * config.rgb_brightness_factor
        img = np.clip(img, 0, 1)

        selected_wavelength = radiance.sel(w=actual_wavelengths[0], method="nearest")
        squeezed = selected_wavelength.squeeze()
        nan_mask = np.isnan(squeezed.values)

        if nan_mask.ndim != 2:
            logger.warning(
                f"Expected 2D nan_mask for RGB, got {nan_mask.ndim}D with shape {nan_mask.shape}. "
                f"Dimensions: {squeezed.dims}"
            )
            nan_mask = None

        img = (img * 255).astype(np.uint8)

        if nan_mask is not None and nan_mask.any():
            if img.shape[:2] == nan_mask.shape:
                img[nan_mask, 0] = 255
                img[nan_mask, 1] = 255
                img[nan_mask, 2] = 255
                logger.info(
                    f"Set {nan_mask.sum()} pixels outside circular FOV to white background"
                )
            else:
                logger.warning(
                    f"Cannot apply white background: img shape {img.shape[:2]} "
                    f"doesn't match nan_mask shape {nan_mask.shape}"
                )

        rgb_path = output_dir / f"{sensor_id}_rgb.png"

        mkdir(output_dir)
        rgb_image = Image.fromarray(img)

        with open_file(rgb_path, "wb") as f:
            rgb_image.save(f, format="PNG")

        logger.info(f"RGB image saved: {rgb_path}")

    def _apply_gaussian_srf(
        self,
        radiance: xr.DataArray,
        fwhm_vnir: float,
        fwhm_swir: float,
        target_wavelengths: Optional[xr.DataArray] = None,
    ) -> xr.DataArray:
        """Apply wavelength-dependent Gaussian SRF convolution.

        Args:
            radiance: Radiance DataArray with wavelength coordinate and bin edges
                     (requires 'bin_wmin' and 'bin_wmax' coordinates)
            fwhm_vnir: FWHM for VNIR wavelengths (<1000nm) in nanometers
            fwhm_swir: FWHM for SWIR wavelengths (>=1000nm) in nanometers
            target_wavelengths: Optional target wavelengths for interpolation.
                               If None, uses radiance wavelengths (default).
                               If provided, applies SRF at each target wavelength.

        Returns:
            Radiance with SRF convolution applied. If target_wavelengths provided,
            output is on target wavelength grid; otherwise, on input wavelength grid.
            Output is 1D with wavelength dimension only (spatial dims averaged out).
        """
        import logging

        logger = logging.getLogger(__name__)

        if target_wavelengths is not None:
            wavelengths = target_wavelengths.values
            logger.info(f"Interpolating to {len(wavelengths)} target wavelengths")
            logger.debug(
                f"Target range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm"
            )
        else:
            if "w" not in radiance.coords:
                logger.warning("No 'w' coordinate found, returning original radiance")
                return radiance
            wavelengths = radiance.w.values

        required_coords = {"bin_wmin", "bin_wmax"}
        if not required_coords.issubset(set(radiance.coords.keys())):
            raise ValueError(
                f"Radiance missing required SRF coordinates: {required_coords}. "
                f"Available: {set(radiance.coords.keys())}. "
                f"Eradiate simulations should include bin edge coordinates."
            )

        import eradiate
        from eradiate.pipelines.logic import apply_spectral_response
        from eradiate.spectral.response import make_gaussian

        logger.debug(f"Applying Gaussian SRF: VNIR={fwhm_vnir}nm, SWIR={fwhm_swir}nm")
        processed_values = []
        for wl in wavelengths:
            fwhm = fwhm_vnir if wl < 1000 else fwhm_swir
            gaussian_srf = make_gaussian(wl, fwhm, pad=True)
            band_srf = eradiate.spectral.BandSRF.from_dataarray(gaussian_srf.srf)
            l_srf = apply_spectral_response(radiance, band_srf)
            processed_values.append(float(l_srf.values.squeeze()))

        data_array = np.array(processed_values, dtype=np.float64)
        if data_array.ndim != 1:
            logger.warning(
                f"SRF output has unexpected shape {data_array.shape}, flattening to 1D"
            )
            data_array = data_array.flatten()

        result = xr.DataArray(
            data=data_array,
            dims=["w"],
            coords={"w": wavelengths},
            attrs=radiance.attrs,
        ).sortby("w")

        return result
