from typing import Dict, Optional

import numpy as np
import xarray as xr

WAVELENGTH_COORD_NAMES = {"w", "wavelength", "wl", "lambda"}


def find_wavelength_coord(da: xr.DataArray) -> Optional[str]:
    """Find the wavelength coordinate in a DataArray.

    Searches for common wavelength coordinate names used in Eradiate
    and other radiative transfer codes.

    Args:
        da: DataArray to search for wavelength coordinate

    Returns:
        Name of wavelength coordinate, or None if not found
    """
    for name in WAVELENGTH_COORD_NAMES:
        if name in da.coords:
            return name
    return None


def normalize_wavelength_coord(da: xr.DataArray, target: str = "w") -> xr.DataArray:
    """Rename wavelength coordinate to a standard name.

    Args:
        da: DataArray with wavelength coordinate
        target: Target coordinate name (default: "w")

    Returns:
        DataArray with renamed wavelength coordinate, or original if
        no wavelength coordinate found or already has target name
    """
    current = find_wavelength_coord(da)
    if current is None or current == target:
        return da
    return da.rename({current: target})


class PostProcessor:
    """Unified post-processor for all sensor types.

    Centralizes all post-processing logic for simulation results.
    Dispatches to sensor-specific processing based on sensor type.

    Currently supported:
    - HYPSTAR: Gaussian SRF convolution + spatial averaging
    """

    def __init__(self, simulation_config):
        """Initialize post-processor.

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

    def process_results(self, results: Dict[str, xr.Dataset]) -> Dict[str, xr.Dataset]:
        """Apply all post-processing to simulation results.

        Iterates through results and applies sensor-specific post-processing
        based on the sensor that produced each result.

        Args:
            results: Dictionary mapping measurement/sensor IDs to xr.Dataset results

        Returns:
            Dictionary with post-processed results
        """
        processed = {}

        for result_id, dataset in results.items():
            sensor = self._find_sensor_for_result(result_id)
            if sensor is None:
                processed[result_id] = dataset
                continue

            processed_ds = self._apply_post_processing(dataset, sensor)
            processed[result_id] = processed_ds

        return processed

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

        Args:
            dataset: Raw simulation result
            sensor: Sensor that produced the result

        Returns:
            Post-processed dataset
        """
        from ..config import GroundInstrumentType, GroundSensor

        if isinstance(sensor, GroundSensor):
            if sensor.instrument == GroundInstrumentType.HYPSTAR:
                return self._apply_hypstar_post_processing(dataset, sensor)

        return dataset

    def _apply_hypstar_post_processing(self, dataset: xr.Dataset, sensor) -> xr.Dataset:
        """Apply HYPSTAR-specific post-processing.

        Applies Gaussian SRF convolution and spatial averaging to simulate
        realistic HYPSTAR instrument response.

        Args:
            dataset: Raw simulation result with radiance data
            sensor: GroundSensor with HYPSTAR instrument

        Returns:
            Post-processed dataset
        """
        config = sensor.hypstar_post_processing
        if config is None:
            return dataset

        result = dataset.copy()

        radiance_var = self._find_radiance_variable(result)
        if radiance_var is None:
            return result

        radiance = result[radiance_var]

        if getattr(config, "spatial_averaging", True):
            radiance = self._apply_spatial_averaging(radiance)

        if getattr(config, "apply_srf", True):
            import logging

            logger = logging.getLogger(__name__)

            fwhm_vnir = getattr(config, "fwhm_vnir_nm", 3.0)
            fwhm_swir = getattr(config, "fwhm_swir_nm", 10.0)

            target_wavelengths = None
            if config.real_reference_file is not None:
                try:
                    logger.info(
                        f"Loading HYPSTAR L2A reference: {config.real_reference_file}"
                    )

                    real_reference_ds = xr.open_dataset(config.real_reference_file)
                    target_wavelengths = real_reference_ds[config.wavelength_variable]
                    logger.info(
                        f"Loaded {len(target_wavelengths)} L2A wavelengths "
                        f"({float(target_wavelengths.min()):.1f} - "
                        f"{float(target_wavelengths.max()):.1f} nm)"
                    )
                except Exception as e:
                    logger.error(f"Failed to load L2A reference: {e}")
                    logger.warning("Falling back to simulation wavelengths")
                    target_wavelengths = None

            radiance = self._apply_gaussian_srf(
                radiance,
                fwhm_vnir=fwhm_vnir,
                fwhm_swir=fwhm_swir,
                target_wavelengths=target_wavelengths,
            )

        result = xr.Dataset(
            {radiance_var: radiance},
            attrs={**result.attrs, "hypstar_post_processed": True},
        )
        return result

    def _find_radiance_variable(self, dataset: xr.Dataset) -> Optional[str]:
        """Find the radiance variable in a dataset.

        Args:
            dataset: Dataset to search

        Returns:
            Name of radiance variable, or None if not found
        """
        for var in ["radiance", "L", "l"]:
            if var in dataset:
                return var
        return None

    def _apply_spatial_averaging(self, radiance: xr.DataArray) -> xr.DataArray:
        """Average over spatial dimensions (FOV pixels).

        Args:
            radiance: Radiance DataArray with spatial dimensions

        Returns:
            Radiance averaged over spatial dimensions
        """
        spatial_dims = [
            d for d in ["x_index", "y_index", "x", "y"] if d in radiance.dims
        ]
        if spatial_dims:
            return radiance.mean(dim=spatial_dims)
        return radiance

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
                               If provided, applies SRF at each target wavelength
                               (use for HYPSTAR L2A validation).

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
            w_coord = find_wavelength_coord(radiance)
            if w_coord is None:
                logger.warning(
                    "No wavelength coordinate found, returning original radiance"
                )
                return radiance
            wavelengths = radiance[w_coord].values
            logger.debug(f"Using {len(wavelengths)} simulation wavelengths")

        required_coords = {"bin_wmin", "bin_wmax"}
        if not required_coords.issubset(set(radiance.coords.keys())):
            raise ValueError(
                f"Radiance missing required SRF coordinates: {required_coords}. "
                f"Available: {set(radiance.coords.keys())}. "
                f"Eradiate simulations should include bin edge coordinates."
            )

        try:
            import eradiate
            from eradiate.pipelines.logic import apply_spectral_response
            from eradiate.spectral.response import make_gaussian
        except ImportError as e:
            logger.error(f"Failed to import eradiate: {e}")
            return radiance

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
