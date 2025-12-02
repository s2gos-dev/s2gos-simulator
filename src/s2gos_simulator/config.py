from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from s2gos_utils import validate_config_version
from s2gos_utils.io.paths import open_file, read_json
from s2gos_utils.io.resolver import resolver
from s2gos_utils.typing import PathLike
from skyfield.api import load, wgs84

from ._version import get_version


class PlatformType(str, Enum):
    """Observation platform types."""

    SATELLITE = "satellite"
    UAV = "uav"
    GROUND = "ground"


class SatellitePlatform(str, Enum):
    """Supported satellite platforms."""

    SENTINEL_2A = "sentinel-2a"
    SENTINEL_2B = "sentinel-2b"
    SENTINEL_3A = "sentinel-3a"
    SENTINEL_3B = "sentinel-3b"
    LANDSAT_8 = "landsat-8"
    LANDSAT_9 = "landsat-9"
    MODIS_TERRA = "modis-terra"
    MODIS_AQUA = "modis-aqua"
    CHIME = "chime"
    ENDMAP = "endmap"
    CUSTOM = "custom"  # For user-defined platforms with custom SRF


class SentinelMSIBand(str, Enum):
    """Sentinel-2 MSI band identifiers."""

    B01 = "1"
    B02 = "2"
    B03 = "3"
    B04 = "4"
    B05 = "5"
    B06 = "6"
    B07 = "7"
    B08 = "8"
    B8A = "8a"
    B09 = "9"
    B10 = "10"
    B11 = "11"
    B12 = "12"


class SentinelOLCIBand(str, Enum):
    """Sentinel-3 OLCI band identifiers."""

    OA01 = "1"
    OA02 = "2"
    OA03 = "3"
    OA04 = "4"
    OA05 = "5"
    OA06 = "6"
    OA07 = "7"
    OA08 = "8"
    OA09 = "9"
    OA10 = "10"
    OA11 = "11"
    OA12 = "12"
    OA13 = "13"
    OA14 = "14"
    OA15 = "15"
    OA16 = "16"
    OA17 = "17"
    OA18 = "18"
    OA19 = "19"
    OA20 = "20"
    OA21 = "21"


class LandsatOLIBand(str, Enum):
    """Landsat OLI band identifiers."""

    B1 = "1"  # Coastal aerosol
    B2 = "2"  # Blue
    B3 = "3"  # Green
    B4 = "4"  # Red
    B5 = "5"  # NIR
    B6 = "6"  # SWIR 1
    B7 = "7"  # SWIR 2
    B8 = "8"  # Panchromatic
    B9 = "9"  # Cirrus


class SatelliteInstrument(str, Enum):
    """Supported satellite instruments."""

    MSI = "msi"  # Sentinel-2 MultiSpectral Instrument
    OLCI = "olci"  # Sentinel-3 Ocean and Land Colour Instrument
    OLI = "oli"  # Landsat Operational Land Imager
    MODIS = "modis"  # MODIS sensor
    VNIR = "vnir"  # Generic VNIR instrument
    HSI = "hsi"  # Hyperspectral instrument
    CUSTOM = "custom"  # For custom instruments


# Platform to instrument mapping
PLATFORM_INSTRUMENTS = {
    SatellitePlatform.SENTINEL_2A: [SatelliteInstrument.MSI],
    SatellitePlatform.SENTINEL_2B: [SatelliteInstrument.MSI],
    SatellitePlatform.SENTINEL_3A: [SatelliteInstrument.OLCI],
    SatellitePlatform.SENTINEL_3B: [SatelliteInstrument.OLCI],
    SatellitePlatform.LANDSAT_8: [SatelliteInstrument.OLI],  # TODO: Implement
    SatellitePlatform.LANDSAT_9: [SatelliteInstrument.OLI],  # TODO: Implement
    SatellitePlatform.MODIS_TERRA: [SatelliteInstrument.MODIS],  # TODO: Implement
    SatellitePlatform.MODIS_AQUA: [SatelliteInstrument.MODIS],  # TODO: Implement
    SatellitePlatform.CHIME: [SatelliteInstrument.VNIR],  # TODO: Implement
    SatellitePlatform.ENDMAP: [SatelliteInstrument.HSI],  # TODO: Implement
    SatellitePlatform.CUSTOM: [SatelliteInstrument.CUSTOM],
}

# Instrument to band mapping
INSTRUMENT_BANDS = {
    SatelliteInstrument.MSI: SentinelMSIBand,
    SatelliteInstrument.OLCI: SentinelOLCIBand,
    SatelliteInstrument.OLI: LandsatOLIBand,  # TODO: Implement
    SatelliteInstrument.MODIS: None,  # TODO: Implement
    SatelliteInstrument.VNIR: None,
    SatelliteInstrument.HSI: None,
    SatelliteInstrument.CUSTOM: None,  # Accept any string for custom
}


class MeasurementType(str, Enum):
    """Radiative quantities that can be measured."""

    BRF = "brf"  # Bidirectional Reflectance Factor
    HDRF = "hdrf"  # Hemispherical-Directional Reflectance Factor
    BHR = "bhr"  # Bi-Hemispherical Reflectance
    BHR_ISO = "bhr_iso"  # Bi-Hemispherical Reflectance (isotropic)
    FLUX_3D = "flux_3d"  # 3D flux distributions
    RADIANCE = "radiance"  # Spectral radiance (instrument-specific)
    IRRADIANCE = "irradiance"  # Spectral irradiance (generic)
    BOA_IRRADIANCE = "boa_irradiance"  # Bottom-of-Atmosphere downward irradiance
    DIGITAL_HEMISPHERICAL_PHOTOGRAPHY = "dhp"  # Digital Hemispherical Photography
    FAPAR = "fapar"  # Fraction of Absorbed Photosynthetically Active Radiation


class ProcessingLevel(str, Enum):
    """Data processing levels."""

    L1B = "l1b"
    L1C = "l1c"  # Orthorectified


class SpectralResponse(BaseModel):
    """Spectral response function configuration."""

    type: Literal["delta", "uniform", "dataset"] = "delta"
    wavelengths: Optional[List[float]] = Field(
        None,
        description="Wavelengths in nm for delta SRF or center wavelength for gaussian SRF",
    )
    wmin: Optional[float] = Field(
        None, description="Minimum wavelength for uniform SRF"
    )
    wmax: Optional[float] = Field(
        None, description="Maximum wavelength for uniform SRF"
    )
    dataset_id: Optional[str] = Field(None, description="Dataset ID for predefined SRF")

    @field_validator("wavelengths")
    @classmethod
    def validate_wavelengths(cls, v):
        if v is not None:
            return [float(w) for w in v]
        return v

    @model_validator(mode="after")
    def validate_srf_config(self):
        if self.type == "delta" and not self.wavelengths:
            raise ValueError("Delta SRF requires wavelengths")
        elif self.type == "uniform" and (not self.wmin or not self.wmax):
            raise ValueError("Uniform SRF requires wmin and wmax")
        elif self.type == "dataset" and not self.dataset_id:
            raise ValueError("Dataset SRF requires dataset_id")
        return self


# TODO: Perhaps it can be less eradiate specific?
SRFType = Union[SpectralResponse, str]


class Illumination(BaseModel):
    """Base illumination configuration."""

    type: str = Field(..., description="Illumination type")
    id: str = Field("illumination", description="Unique identifier")


class DirectionalIllumination(Illumination):
    """Directional illumination (e.g., sun)."""

    type: Literal["directional"] = "directional"
    zenith: float = Field(
        30.0, ge=0.0, le=90.0, description="Solar zenith angle in degrees"
    )
    azimuth: float = Field(
        180.0, ge=0.0, lt=360.0, description="Solar azimuth angle in degrees"
    )
    irradiance_dataset: str = Field(
        "thuillier_2003", description="Solar irradiance dataset"
    )

    @classmethod
    def from_date_and_location(
        cls,
        time: datetime,
        latitude: float,
        longitude: float,
        irradiance_dataset: str = "thuillier_2003",
    ) -> "DirectionalIllumination":
        """
        Creates a DirectionalIllumination instance by calculating solar angles
        for a given time and location, and converting them to Eradiate conventions.

        Args:
            time: The date and time of the observation.
            latitude: Observer's latitude in degrees.
            longitude: Observer's longitude in degrees.
            irradiance_dataset: Name of the solar irradiance dataset to use.

        Returns:
            A new DirectionalIllumination instance with corrected zenith and azimuth.

        Raises:
            ValueError: If the sun is below the horizon at the specified time.
        """
        # Load timescale and ephemeris
        ts = load.timescale()
        ephemeris_path = str(resolver.resolve("de421.bsp", strict=False))
        planets = load(ephemeris_path)

        # Convert datetime to skyfield time
        skyfield_time = ts.utc(
            time.year, time.month, time.day, time.hour, time.minute, time.second
        )

        # Define observer location
        earth = planets["earth"]
        location = earth + wgs84.latlon(latitude, longitude)

        # Calculate sun position
        sun = planets["sun"]
        astrometric = location.at(skyfield_time).observe(sun)
        apparent = astrometric.apparent()
        alt, az, _ = apparent.altaz()

        # Check if sun is above horizon
        if alt.degrees < 0:
            raise ValueError(
                f"Sun is below horizon (altitude: {alt.degrees:.2f}°) at the specified time"
            )

        # Convert to zenith angle (degrees)
        zenith_angle = 90.0 - alt.degrees

        # Convert azimuth to Eradiate convention
        # Skyfield: 0°=North, 90°=East, 180°=South, 270°=West
        # Eradiate: 0°=East, 90°=North, 180°=West, 270°=South
        eradiate_az = (90.0 - az.degrees) % 360.0

        return cls(
            zenith=zenith_angle,
            azimuth=eradiate_az,
            irradiance_dataset=irradiance_dataset,
        )


class ConstantIllumination(Illumination):
    """Constant uniform illumination."""

    type: Literal["constant"] = "constant"
    radiance: float = Field(1.0, gt=0.0, description="Constant radiance value")


class BaseViewing(BaseModel):
    """Base class for viewing configurations."""

    type: str


class AngularViewing(BaseViewing):
    """
    Viewing defined by zenith and azimuth angles, typically relative to a target.

    This is ideal for distant sensors (satellites)
    """

    type: Literal["angular"] = "angular"
    zenith: float = Field(
        0.0,
        ge=0.0,
        le=180.0,
        description="Viewing zenith angle (0=nadir, 90=horizon, 180=upward)",
    )
    azimuth: float = Field(0.0, ge=0.0, lt=360.0, description="Viewing azimuth angle")
    target: Optional[List[float]] = Field(
        None,
        description="Center of the observed area [x, y, z]. If None, defaults to scene center [0,0,0].",
    )


class AngularFromOriginViewing(BaseViewing):
    """
    Viewing defined by an origin position and zenith/azimuth angles from that origin.

    This is ideal for radiancemeters and cameras where you know the sensor position
    and want to specify the pointing direction with angles.
    """

    type: Literal["angular_from_origin"] = "angular_from_origin"
    origin: List[float] = Field(
        ..., description="3D sensor position [x, y, z] in meters"
    )
    zenith: float = Field(
        0.0,
        ge=0.0,
        le=180.0,
        description="Pointing zenith angle (0=down/nadir, 90=horizon, 180=up)",
    )
    azimuth: float = Field(
        0.0, ge=0.0, lt=360.0, description="Pointing azimuth angle (0=East, 90=North)"
    )
    up: Optional[List[float]] = Field(
        [0, 0, 1], description="Up direction for cameras [x, y, z] (default Z-up)"
    )


class LookAtViewing(BaseViewing):
    """
    Viewing defined by an origin and a target point.

    This is ideal for UAVs or perspective cameras where the exact position
    of the sensor and its target are known.
    """

    type: Literal["directional"] = "directional"
    origin: List[float] = Field(
        ..., description="3D sensor position [x, y, z] in meters"
    )
    target: List[float] = Field(
        [0.0, 0.0, 0.0], description="3D target position [x, y, z] in meters"
    )
    up: Optional[List[float]] = Field(
        [0, 0, 1], description="Up direction for cameras [x, y, z] (default Z-up)"
    )


class HemisphericalViewing(BaseViewing):
    """
    Viewing that covers the entire upper or lower hemisphere.
    """

    type: Literal["hemispherical"] = "hemispherical"
    origin: List[float] = Field(
        ..., description="3D sensor position [x, y, z] in meters."
    )
    upward_looking: bool = Field(
        True,
        description="True for upward-looking (sky), False for downward-looking (ground)",
    )


ViewingType = Union[
    AngularViewing, AngularFromOriginViewing, LookAtViewing, HemisphericalViewing
]


class RadiativeQuantityConfig(BaseModel):
    """Configuration for radiative quantities independent of sensors."""

    id: Optional[str] = Field(
        None, description="Unique identifier for this radiative quantity measurement"
    )
    quantity: MeasurementType = Field(
        ..., description="Type of radiative quantity to calculate"
    )
    srf: Optional[SRFType] = Field(None, description="Spectral response function")
    viewing_zenith: Optional[float] = Field(
        None,
        ge=0.0,
        le=180.0,
        description="Viewing zenith angle for directional quantities",
    )
    viewing_azimuth: Optional[float] = Field(
        None,
        ge=0.0,
        lt=360.0,
        description="Viewing azimuth angle for directional quantities",
    )
    target_lat: float = (
        Field(
            0,
            description="Viewing azimuth angle for directional quantities",
        ),
    )
    target_lon: float = (
        Field(
            0,
            description="Viewing azimuth angle for directional quantities",
        ),
    )
    samples_per_pixel: int = Field(
        64, ge=1, description="Number of samples per pixel for Monte Carlo calculations"
    )
    reference_panel_offset_m: Optional[float] = Field(
        None,
        ge=0.0,
        description="Height offset above terrain for HDRF reference panel (accounts for vegetation height). Required for HDRF measurements.",
    )

    @model_validator(mode="after")
    def validate_quantity_config(self):
        """Validate configuration based on quantity type."""
        if self.quantity in [MeasurementType.BRF, MeasurementType.HDRF]:
            if self.viewing_zenith is None or self.viewing_azimuth is None:
                raise ValueError(
                    f"{self.quantity} requires viewing_zenith and viewing_azimuth"
                )

        if self.quantity == MeasurementType.HDRF:
            if self.reference_panel_offset_m is None:
                raise ValueError(
                    "HDRF measurements require explicit reference_panel_offset_m "
                    "(height above terrain for reference panel placement)"
                )
            if self.target_lat is None or self.target_lon is None:
                raise ValueError("HDRF measurements require target_lat and target_lon")

        return self


class IrradianceMeasurementConfig(BaseModel):
    """Configuration for BOA irradiance measurements at specific locations.

    Uses the white reference disk technique to measure downward irradiance at BOA:
    - Places a small white Lambertian disk (ρ=1.0) at the measurement location
    - Measures radiance from the disk
    - Converts to irradiance: E = π × L
    - Averages multiple samples to reduce Monte Carlo noise

    This is the same technique used for HDRF reference measurements, now
    available as a standalone measurement for any location.
    """

    id: str = Field(
        ..., description="Unique identifier for this irradiance measurement"
    )
    target_lat: float = Field(
        ...,
        description="Target latitude for irradiance measurement (WGS84 decimal degrees)",
    )
    target_lon: float = Field(
        ...,
        description="Target longitude for irradiance measurement (WGS84 decimal degrees)",
    )
    height_offset_m: float = Field(
        0.0,
        ge=0.0,
        description="Height above terrain for measurement (accounts for vegetation/sensor height)",
    )
    srf: Optional[SRFType] = Field(
        None, description="Spectral response function for irradiance measurement"
    )
    samples_per_pixel: int = Field(
        512,
        ge=1,
        description="Number of samples per pixel for Monte Carlo calculations (higher = less noise)",
    )

    @model_validator(mode="after")
    def validate_irradiance_config(self):
        """Validate irradiance measurement configuration."""
        import logging

        logger = logging.getLogger(__name__)

        # Validate lat/lon ranges
        if not -90 <= self.target_lat <= 90:
            raise ValueError(
                f"target_lat must be in range [-90, 90], got {self.target_lat}"
            )
        if not -180 <= self.target_lon <= 180:
            raise ValueError(
                f"target_lon must be in range [-180, 180], got {self.target_lon}"
            )

        # Validate height offset
        if self.height_offset_m < 0:
            raise ValueError(
                f"height_offset_m must be >= 0, got {self.height_offset_m}. "
                "Negative heights (underground) are not physical."
            )

        # Warning for large height offsets
        if self.height_offset_m > 100:
            logger.warning(
                f"Large height_offset_m ({self.height_offset_m}m) detected for '{self.id}'. "
                "Ensure this is intentional - typical ground sensors are <50m."
            )

        # Validate samples_per_pixel
        if self.samples_per_pixel < 32:
            logger.warning(
                f"Low samples_per_pixel ({self.samples_per_pixel}) for '{self.id}'. "
                "Recommend >= 128 for acceptable noise levels, >= 512 for low noise."
            )

        return self


class BaseSensor(BaseModel):
    """Base sensor configuration."""

    id: Optional[str] = Field(
        None, description="Unique sensor identifier (auto-generated if not provided)"
    )
    platform_type: PlatformType
    viewing: ViewingType
    srf: Optional[SRFType] = Field(None, description="Spectral response function")
    produces: List[MeasurementType] = Field(
        [MeasurementType.RADIANCE],
        description="List of radiative quantities to be produced by this sensor configuration",
    )
    samples_per_pixel: int = Field(64, ge=1)
    noise_model: Optional[Dict[str, Any]] = Field(None)


class SatelliteSensor(BaseSensor):
    """Satellite sensor configuration with strong validation."""

    platform_type: Literal[PlatformType.SATELLITE] = PlatformType.SATELLITE
    viewing: AngularViewing
    platform: SatellitePlatform
    instrument: SatelliteInstrument
    band: str
    film_resolution: Tuple[int, int] = Field(
        ..., description="Pixel grid dimensions (width, height) for 2D imaging"
    )
    target_center_lat: float = Field(
        ..., description="Target center latitude in WGS84 decimal degrees"
    )
    target_center_lon: float = Field(
        ..., description="Target center longitude in WGS84 decimal degrees"
    )
    target_size_km: Union[float, Tuple[float, float]] = Field(
        ...,
        description="Target area size: float for square (km), tuple for rectangular (width_km, height_km)",
    )

    @model_validator(mode="after")
    def validate_and_set_defaults(self):
        """Validate platform/instrument/band combination and set defaults."""

        if self.platform != SatellitePlatform.CUSTOM:
            valid_instruments = PLATFORM_INSTRUMENTS.get(self.platform, [])
            if self.instrument not in valid_instruments:
                raise ValueError(
                    f"Instrument '{self.instrument.value}' is not valid for platform '{self.platform.value}'. "
                    f"Valid instruments: {[inst.value for inst in valid_instruments]}"
                )

        if (
            self.platform != SatellitePlatform.CUSTOM
            and self.instrument != SatelliteInstrument.CUSTOM
        ):
            band_enum = INSTRUMENT_BANDS.get(self.instrument)
            if band_enum is not None:
                try:
                    valid_band = band_enum(self.band)
                    self.band = valid_band.value
                except ValueError:
                    valid_bands = [band.value for band in band_enum]
                    raise ValueError(
                        f"Band '{self.band}' is not valid for instrument '{self.instrument.value}'. "
                        f"Valid bands: {valid_bands}"
                    )

        if self.platform == SatellitePlatform.CUSTOM and self.srf is None:
            raise ValueError(
                "Custom platform requires an explicit SRF. "
                "Please provide an SRF using the 'srf' parameter."
            )

        if self.viewing.target is None:
            self.viewing.target = [0.0, 0.0, 0.0]

        # TODO: Could be less eradiate specific?
        if self.srf is None and self.platform != SatellitePlatform.CUSTOM:
            platform_norm = self.platform.value.replace("-", "_")
            self.srf = f"{platform_norm}-{self.instrument.value}-{self.band}"

        if self.id is None:
            if self.platform == SatellitePlatform.CUSTOM:
                base_id = f"custom_{self.instrument.value}_{self.band}"
            else:
                platform_clean = self.platform.value.replace("-", "_")
                base_id = f"{platform_clean}_{self.instrument.value}_{self.band}"

            if self.viewing.zenith > 0:
                self.id = f"{base_id}_oblique{int(self.viewing.zenith)}"
            else:
                self.id = f"{base_id}_nadir"

        # Validate film resolution limits
        width, height = self.film_resolution
        if width <= 0 or height <= 0:
            raise ValueError("Film resolution dimensions must be positive integers")
        if width > 2048 or height > 2048:
            raise ValueError(
                f"Film resolution {self.film_resolution} exceeds maximum of 2048x2048 pixels. "
                "Consider reducing resolution for memory and performance reasons."
            )

        return self

    @property
    def pixel_size_m(self) -> Tuple[float, float]:
        """
        Calculate ground pixel size in meters.

        Returns:
            Tuple of (pixel_size_x_m, pixel_size_y_m) in meters per pixel
        """
        if isinstance(self.target_size_km, (int, float)):
            # Square area
            width_km = height_km = self.target_size_km
        else:
            # Rectangular area
            width_km, height_km = self.target_size_km

        pixel_size_x = (width_km * 1000) / self.film_resolution[0]
        pixel_size_y = (height_km * 1000) / self.film_resolution[1]

        return pixel_size_x, pixel_size_y


class UAVInstrumentType(str, Enum):
    """Enum for UAV-mounted instrument types."""

    PERSPECTIVE_CAMERA = "perspective_camera"
    RADIANCEMETER = "radiancemeter"


class UAVSensor(BaseSensor):
    platform_type: Literal[PlatformType.UAV] = PlatformType.UAV
    instrument: UAVInstrumentType
    viewing: Union[LookAtViewing, AngularFromOriginViewing]
    fov: Optional[float] = Field(None)
    resolution: Optional[List[int]] = Field(None)
    terrain_relative_height: bool = Field(
        False,
        description="If True, z-coordinates are interpreted as offsets from terrain elevation. "
        "For LookAtViewing, both origin and target z-coordinates are terrain-relative. "
        "For AngularFromOriginViewing, only origin z-coordinate is terrain-relative. "
        "If False (default), all z-coordinates are absolute elevations.",
    )

    @model_validator(mode="after")
    def validate_and_set_defaults(self):
        if self.instrument == UAVInstrumentType.PERSPECTIVE_CAMERA:
            if not self.fov or not self.resolution:
                print(
                    "Warning: 'fov' and 'resolution' are recommended for a perspective_camera."
                )
        elif self.instrument == UAVInstrumentType.RADIANCEMETER:
            if self.fov or self.resolution:
                print(
                    "Warning: 'fov' and 'resolution' are ignored for a 'radiancemeter'."
                )

        if self.id is None:
            self.id = f"uav_{self.instrument.value}_{int(self.viewing.origin[2])}m"
        return self


class GroundInstrumentType(str, Enum):
    """Enum for ground instrument types for robustness."""

    HYPSTAR = "hypstar"
    PERSPECTIVE_CAMERA = "perspective_camera"
    PYRANOMETER = "pyranometer"
    FLUX_METER = "flux_meter"
    DHP_CAMERA = "dhp_camera"
    RADIANCEMETER = "radiancemeter"


class GroundSensor(BaseSensor):
    platform_type: Literal[PlatformType.GROUND] = PlatformType.GROUND
    instrument: GroundInstrumentType
    viewing: Union[LookAtViewing, AngularFromOriginViewing]
    fov: Optional[float] = Field(
        None,
        description="Field of view in degrees (for camera-like instruments: HYPSTAR, perspective_camera, dhp_camera)",
    )
    resolution: Optional[List[int]] = Field(
        None,
        description="Film resolution [width, height] (for camera-like instruments: HYPSTAR, perspective_camera, dhp_camera)",
    )
    terrain_relative_height: bool = Field(
        False,
        description="If True, z-coordinates are interpreted as offsets from terrain elevation. "
        "For LookAtViewing, both origin and target z-coordinates are terrain-relative. "
        "For AngularFromOriginViewing, only origin z-coordinate is terrain-relative. "
        "If False (default), all z-coordinates are absolute elevations.",
    )

    @model_validator(mode="after")
    def set_defaults_and_validate(self):
        if self.instrument in [
            GroundInstrumentType.HYPSTAR,
            GroundInstrumentType.PERSPECTIVE_CAMERA,
            GroundInstrumentType.DHP_CAMERA,
        ]:
            if not isinstance(self.viewing, (LookAtViewing, AngularFromOriginViewing)):
                raise ValueError(
                    f"'{self.instrument.value}' requires a pointing view "
                    f"('directional' or 'angular_from_origin'), not '{self.viewing.type}'."
                )

            # Warn if camera-like instruments don't have fov/resolution specified
            if self.instrument == GroundInstrumentType.HYPSTAR:
                if self.fov is None or self.resolution is None:
                    print(
                        f"Warning: HYPSTAR sensor '{self.id or 'unnamed'}' does not have 'fov' "
                        f"and/or 'resolution' specified. Using defaults: fov=5.0°, resolution=[5, 5]"
                    )

        elif self.instrument in [
            GroundInstrumentType.PYRANOMETER,
            GroundInstrumentType.FLUX_METER,
        ]:
            if not isinstance(self.viewing, HemisphericalViewing):
                raise ValueError(
                    f"'{self.instrument.value}' requires 'hemispherical' viewing, "
                    f"not '{self.viewing.type}'."
                )

        if self.id is None:
            self.id = f"ground_{self.instrument.value}_{int(self.viewing.origin[2])}m"
        return self


class ProcessingConfig(BaseModel):
    """Processing configuration."""

    orthorectified: bool = Field(
        False, description="Whether to produce orthorectified output"
    )


def create_satellite_sensor(
    platform: SatellitePlatform,
    instrument: SatelliteInstrument,
    band: str,
    zenith: float = 0.0,
    azimuth: float = 0.0,
    **kwargs,
) -> SatelliteSensor:
    """Create a satellite sensor with validated platform/instrument/band combination.

    Args:
        platform: Satellite platform (use SatellitePlatform enum)
        instrument: Satellite instrument (use SatelliteInstrument enum)
        band: Band identifier (validated against instrument)
        zenith: Viewing zenith angle in degrees (0=nadir, >0=oblique)
        azimuth: Viewing azimuth angle in degrees
        **kwargs: Additional sensor parameters

    Returns:
        SatelliteSensor configured with the specified identifiers
    """
    return SatelliteSensor(
        platform=platform,
        instrument=instrument,
        band=band,
        viewing=AngularViewing(zenith=zenith, azimuth=azimuth),
        **kwargs,
    )


def create_custom_satellite_sensor(
    instrument_name: str,
    band_name: str,
    srf: SRFType,
    zenith: float = 0.0,
    azimuth: float = 0.0,
    **kwargs,
) -> SatelliteSensor:
    """Create a custom satellite sensor with user-provided SRF.

    Args:
        instrument_name: Custom instrument identifier
        band_name: Custom band identifier
        srf: Spectral response function (SpectralResponse object or string)
        zenith: Viewing zenith angle in degrees (0=nadir, >0=oblique)
        azimuth: Viewing azimuth angle in degrees
        **kwargs: Additional sensor parameters

    Returns:
        SatelliteSensor configured as a custom platform
    """
    return SatelliteSensor(
        platform=SatellitePlatform.CUSTOM,
        instrument=SatelliteInstrument.CUSTOM,
        band=band_name,
        srf=srf,
        viewing=AngularViewing(zenith=zenith, azimuth=azimuth),
        **kwargs,
    )


def create_uav_sensor(
    camera_type: str,
    altitude: float = 100.0,
    origin: Optional[List[float]] = None,
    target: Optional[List[float]] = None,
    **kwargs,
) -> UAVSensor:
    """Create a UAV sensor with flexible positioning.

    Args:
        camera_type: Camera type (e.g., "RGB", "hyperspectral")
        origin: UAV position [x, y, z] in meters (defaults to [0, 0, altitude])
        target: Target position [x, y, z] in meters (defaults to [0, 0, 0])
        **kwargs: Additional sensor parameters (zenith, azimuth, fov, resolution, etc.)

    Returns:
        UAVSensor configured with the specified parameters
    """
    return UAVSensor(camera_type=camera_type, origin=origin, target=target, **kwargs)


def create_ground_sensor(
    instrument: str,
    height: float = 2.0,
    position: Optional[List[float]] = None,
    **kwargs,
) -> GroundSensor:
    """Create a ground-based sensor with flexible viewing.

    Args:
        instrument: Instrument identifier (e.g., "HYPSTAR", "pyranometer", "flux_meter", "perspective_camera")
        height: Height above ground in meters
        position: 3D position [x, y, z] in meters (defaults to [0, 0, height])
        **kwargs: Additional sensor parameters (upward_looking, target, zenith, azimuth, hemispherical, etc.)

    Returns:
        GroundSensor configured with the specified parameters
    """
    if position is None:
        position = [0.0, 0.0, height]

    return GroundSensor(
        instrument=instrument, height=height, position=position, **kwargs
    )


class SimulationConfig(BaseModel):
    """
    Comprehensive simulation configuration containing everything needed to run a simulation.

    This single configuration class includes:
    - Sensor definitions for all platforms
    - Illumination settings
    - Measurement types and processing levels
    - Noise modeling and validation
    """

    # Metadata
    config_version: str = Field(
        default_factory=get_version, description="Configuration schema version"
    )
    name: str = Field(..., description="Simulation name")
    description: Optional[str] = Field(None, description="Simulation description")
    created_at: datetime = Field(default_factory=datetime.now)

    # Core configuration
    illumination: Union[DirectionalIllumination, ConstantIllumination] = Field(
        default_factory=None, description="Illumination configuration"
    )
    sensors: List[Union[SatelliteSensor, UAVSensor, GroundSensor]] = Field(
        default_factory=list, description="List of sensors to simulate"
    )
    radiative_quantities: List[RadiativeQuantityConfig] = Field(
        default_factory=list,
        description="List of radiative quantities to calculate independently",
    )
    irradiance_measurements: List[IrradianceMeasurementConfig] = Field(
        default_factory=list,
        description="List of BOA irradiance measurements at specific locations",
    )

    # Processing configuration
    processing: ProcessingConfig = Field(
        default_factory=ProcessingConfig, description="Processing configuration"
    )

    # Advanced options
    enable_noise: bool = Field(False, description="Enable noise modeling")

    backend_hints: Dict[str, Any] = Field(
        default_factory=dict,
        description="""Backend-specific configuration hints (e.g., {'eradiate': {'mode': 'ckd_double'}}).

        For Eradiate backend:
        - 'mode': Spectral mode ('mono', 'ckd', 'mono_polarized', 'ckd_polarized', etc.)
        """,
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "json_encoders": {datetime: lambda v: v.isoformat()},
    }

    @field_validator("sensors")
    @classmethod
    def validate_sensors(cls, v):
        """Validate sensor list."""
        if v:
            sensor_ids = [sensor.id for sensor in v]
            if len(sensor_ids) != len(set(sensor_ids)):
                raise ValueError("Sensor IDs must be unique")
        return v

    @model_validator(mode="after")
    def validate_simulation_config(self):
        """Validate simulation configuration."""
        if (
            not self.sensors
            and not self.radiative_quantities
            and not self.irradiance_measurements
        ):
            raise ValueError(
                "At least one sensor, radiative quantity, or irradiance measurement must be specified"
            )

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    def to_json(self, path: Optional[PathLike] = None, indent: int = 2) -> str:
        """Export to JSON format."""
        json_str = self.model_dump_json(indent=indent)
        if path:
            with open_file(path, "w") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, path: PathLike) -> "SimulationConfig":
        """Load from JSON file with version compatibility checking."""
        data = read_json(path)

        # Validate configuration version
        validated_data = validate_config_version(
            "simulation_config", data, get_version(), "simulation configuration"
        )

        return cls(**validated_data)

    def add_sensor(
        self, sensor: Union[SatelliteSensor, UAVSensor, GroundSensor]
    ) -> None:
        """Add a sensor to the configuration."""
        if sensor.id in [s.id for s in self.sensors]:
            raise ValueError(f"Sensor ID '{sensor.id}' already exists")
        self.sensors.append(sensor)

    def remove_sensor(self, sensor_id: str) -> None:
        """Remove a sensor by ID."""
        self.sensors = [s for s in self.sensors if s.id != sensor_id]

    def get_sensor(
        self, sensor_id: str
    ) -> Optional[Union[SatelliteSensor, UAVSensor, GroundSensor]]:
        """Get a sensor by ID."""
        for sensor in self.sensors:
            if sensor.id == sensor_id:
                return sensor
        return None

    def get_sensors_by_platform(
        self, platform: PlatformType
    ) -> List[Union[SatelliteSensor, UAVSensor, GroundSensor]]:
        """Get all sensors for a specific platform."""
        return [s for s in self.sensors if s.platform_type == platform]

    def get_nadir_sensors(
        self,
    ) -> List[Union[SatelliteSensor, UAVSensor, GroundSensor]]:
        """Get all sensors with nadir viewing."""
        nadir_sensors = []
        for sensor in self.sensors:
            if hasattr(sensor.viewing, "zenith") and sensor.viewing.zenith == 0.0:
                nadir_sensors.append(sensor)
        return nadir_sensors

    def get_oblique_sensors(
        self,
    ) -> List[Union[SatelliteSensor, UAVSensor, GroundSensor]]:
        """Get all sensors with oblique viewing."""
        oblique_sensors = []
        for sensor in self.sensors:
            if hasattr(sensor.viewing, "zenith") and sensor.viewing.zenith > 0.0:
                oblique_sensors.append(sensor)
        return oblique_sensors

    def get_sensors_by_measurement(
        self, measurement: MeasurementType
    ) -> List[Union[SatelliteSensor, UAVSensor, GroundSensor]]:
        """Get all sensors for a specific measurement type."""
        matching_sensors = []
        for sensor in self.sensors:
            if hasattr(sensor, "produces") and measurement in sensor.produces:
                matching_sensors.append(sensor)
        return matching_sensors

    @property
    def output_quantities(self) -> List[MeasurementType]:
        """Get output quantities based on sensors."""
        # In the new system, sensors specify what they produce in the 'produces' field
        quantities = []
        for sensor in self.sensors:
            if hasattr(sensor, "produces"):
                quantities.extend(sensor.produces)
            else:
                # Default to radiance for all sensors
                quantities.append(MeasurementType.RADIANCE)
        return list(set(quantities))

    @property
    def wavelength_range(self) -> tuple:
        """Get wavelength range based on sensors."""
        min_wl, max_wl = float("inf"), 0.0

        for sensor in self.sensors:
            if isinstance(sensor.srf, SpectralResponse):
                if sensor.srf.type == "delta" and sensor.srf.wavelengths:
                    min_wl = min(min_wl, min(sensor.srf.wavelengths))
                    max_wl = max(max_wl, max(sensor.srf.wavelengths))
                elif (
                    sensor.srf.type == "uniform" and sensor.srf.wmin and sensor.srf.wmax
                ):
                    min_wl = min(min_wl, sensor.srf.wmin)
                    max_wl = max(max_wl, sensor.srf.wmax)

        if min_wl == float("inf"):
            return (400.0, 2500.0)

        return (min_wl, max_wl)

    def validate_configuration(self) -> List[str]:
        """Validate the complete configuration and return any errors."""
        errors = []

        if not self.sensors:
            errors.append("No sensors defined")

        sensor_ids = [sensor.id for sensor in self.sensors]
        if len(sensor_ids) != len(set(sensor_ids)):
            errors.append("Sensor IDs must be unique")

        return errors
