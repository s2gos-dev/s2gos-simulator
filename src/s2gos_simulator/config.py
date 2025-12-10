from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from s2gos_utils import validate_config_version
from s2gos_utils.io.paths import open_file, read_json
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


# MeasurementType enum removed - replaced with discriminated union configs below


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
        planets = load("de421.bsp")

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
    terrain_relative_height: bool = Field(
        default=False,
        description=(
            "If True, z-coordinate in origin is offset from terrain surface. "
            "Backend will query DEM elevation at (x, y) and add z offset. "
            "If False (default), z-coordinate is absolute elevation in scene coordinate system."
        ),
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
    terrain_relative_height: bool = Field(
        default=False,
        description=(
            "If True, z-coordinates in origin and target are offsets from terrain surface. "
            "Backend will query DEM elevation at (x, y) and add z offset. "
            "If False (default), z-coordinates are absolute elevations in scene coordinate system."
        ),
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
    terrain_relative_height: bool = Field(
        default=False,
        description=(
            "If True, z-coordinate in origin is offset from terrain surface. "
            "Backend will query DEM elevation at (x, y) and add z offset. "
            "If False (default), z-coordinate is absolute elevation in scene coordinate system."
        ),
    )


# LocationBased class removed - was unused duplicate of HemisphericalMeasurementLocation
# Use HemisphericalMeasurementLocation for location-based measurements instead

ViewingType = Union[
    AngularViewing,
    AngularFromOriginViewing,
    LookAtViewing,
    HemisphericalViewing,
    # LocationBased removed - use HemisphericalMeasurementLocation for location-based measurements
]


class IrradianceConfig(BaseModel):
    """Configuration for BOA irradiance measurement using white reference disk.

    Measures downward hemispheric irradiance at bottom-of-atmosphere using the
    white reference disk technique:
    - Places a small white Lambertian disk (ρ=1.0) at the measurement location
    - Measures upward radiance from the disk
    - Converts to downward irradiance: E = π × L
    - Averages samples to reduce Monte Carlo noise

    This is the fundamental irradiance measurement used as reference for HDRF.
    """

    type: Literal["irradiance"] = "irradiance"
    id: str = Field(description="Unique identifier for this irradiance measurement")
    location: HemisphericalMeasurementLocation
    samples_per_pixel: int = 512


class BRFConfig(BaseModel):
    """Configuration for Bidirectional Reflectance Factor (BRF) measurement.

    BRF is the ratio of radiance reflected in a specific direction to that from
    a perfect Lambertian reflector under the same illumination conditions.

    Two modes:
    1. Reference mode: Specify radiance_sensor_id + reference_radiance_sensor_id
    2. Auto-generation mode: Specify viewing angles, backend creates sensors
    """

    type: Literal["brf"] = "brf"
    id: Optional[str] = Field(None, description="Unique identifier")

    # Mode 1: Reference existing sensors
    radiance_sensor_id: Optional[str] = Field(
        None, description="ID of sensor providing actual radiance measurement"
    )
    reference_radiance_sensor_id: Optional[str] = Field(
        None, description="ID of sensor providing reference radiance (Lambertian disk)"
    )

    # Mode 2: Auto-generation mode
    viewing_zenith: Optional[float] = Field(
        None, ge=0.0, le=180.0, description="Viewing zenith angle (auto-generation)"
    )
    viewing_azimuth: Optional[float] = Field(
        None, ge=0.0, lt=360.0, description="Viewing azimuth angle (auto-generation)"
    )

    @model_validator(mode="after")
    def validate_brf_mode(self):
        """Validate that either sensor references OR viewing angles are provided."""
        has_refs = (
            self.radiance_sensor_id is not None
            and self.reference_radiance_sensor_id is not None
        )
        has_autogen = (
            self.viewing_zenith is not None and self.viewing_azimuth is not None
        )

        if has_refs and has_autogen:
            raise ValueError(
                "Specify either sensor references OR viewing angles, not both"
            )

        if not has_refs and not has_autogen:
            raise ValueError(
                "Must specify either references (radiance_sensor_id + reference_radiance_sensor_id) "
                "OR auto-generation mode (viewing_zenith + viewing_azimuth)"
            )

        return self


class HDRFConfig(BaseModel):
    """Configuration for Hemispherical-Directional Reflectance Factor (HDRF).

    HDRF measures the ratio of radiance to BOA irradiance.

    Two modes:
    1. Reference mode: Specify radiance_sensor_id + irradiance_measurement_id
    2. Auto-generation mode: Specify viewing geometry, backend creates sensor + measurement
    """

    type: Literal["hdrf"] = "hdrf"
    id: Optional[str] = Field(None, description="Unique identifier")

    # Mode 1: Reference existing sensor + measurement
    radiance_sensor_id: Optional[str] = Field(
        None, description="ID of sensor providing radiance measurement"
    )
    irradiance_measurement_id: Optional[str] = Field(
        None, description="ID of IrradianceConfig providing BOA irradiance"
    )

    # Mode 2: Auto-generation mode
    instrument: Optional[Literal["radiancemeter", "hemispherical"]] = Field(
        None, description="Instrument type for auto-generation"
    )
    location: Optional[HemisphericalMeasurementLocation] = Field(
        None, description="Location for hemispherical HDRF (auto-generation)"
    )
    viewing: Optional[Union[LookAtViewing, AngularFromOriginViewing]] = Field(
        None, description="Viewing geometry for radiancemeter HDRF (auto-generation)"
    )
    reference_height_offset_m: Optional[float] = Field(
        None,
        description="Height offset for reference irradiance disk (auto-generation)",
    )
    srf: Optional[SRFType] = Field(None, description="SRF for auto-generated sensors")
    samples_per_pixel: int = Field(
        64, description="Samples per pixel for auto-generated sensors"
    )
    terrain_relative_height: bool = Field(
        False, description="Terrain-relative height interpretation"
    )

    @model_validator(mode="after")
    def validate_hdrf_mode(self):
        """Validate that either sensor references OR auto-generation specs are provided."""
        has_refs = (
            self.radiance_sensor_id is not None
            and self.irradiance_measurement_id is not None
        )
        has_autogen = self.instrument is not None

        if has_refs and has_autogen:
            raise ValueError(
                "Specify either sensor/measurement references OR auto-generation mode, not both"
            )

        if not has_refs and not has_autogen:
            raise ValueError(
                "Must specify either references (radiance_sensor_id + irradiance_measurement_id) "
                "OR auto-generation mode (instrument + location/viewing)"
            )

        # Validate auto-generation mode
        if has_autogen:
            if self.instrument == "hemispherical" and not self.location:
                raise ValueError("instrument='hemispherical' requires 'location'")
            if self.instrument == "radiancemeter" and not self.viewing:
                raise ValueError("instrument='radiancemeter' requires 'viewing'")
            if not self.reference_height_offset_m:
                raise ValueError(
                    "Auto-generation mode requires 'reference_height_offset_m'"
                )

        return self


class HCRFPostProcessingConfig(BaseModel):
    """Post-processing configuration for HCRF measurements.

    HCRF produces images (wavelength × x_index × y_index) which can be
    post-processed in various ways before final analysis.
    """

    # Spectral response function convolution
    apply_srf_convolution: bool = Field(
        False,
        description="Apply Gaussian SRF convolution to spectral data",
    )
    srf_fwhm_nm: Optional[float] = Field(
        None,
        gt=0.0,
        description="Full Width at Half Maximum for Gaussian SRF (nm). Common values: HYPSTAR VNIR=3.0, SWIR=10.0",
    )

    # Spatial averaging over FOV
    compute_spatial_average: bool = Field(
        True,
        description="Average HCRF values over spatial dimensions (FOV pixels)",
    )
    spatial_statistic: Literal["mean", "median"] = Field(
        "mean",
        description="Statistic to use for spatial averaging",
    )

    # RGB visualization
    create_rgb_image: bool = Field(
        False,
        description="Generate RGB image from spectral data",
    )
    rgb_wavelengths_nm: Tuple[float, float, float] = Field(
        (660.0, 550.0, 440.0),
        description="Wavelengths (nm) to use for RGB channels [R, G, B]",
    )
    rgb_scaling_factors: Tuple[float, float, float] = Field(
        (1.0, 1.0, 1.0),
        description="Scaling factors for RGB channels",
    )


class HCRFConfig(BaseModel):
    """Configuration for Hemispherical-Conical Reflectance Factor (HCRF).

    HCRF is a **purely ideal, theoretical measurement** that captures the ratio
    of radiance within a conical field of view to BOA irradiance.

    Two modes:
    1. Reference mode: Specify radiance_sensor_id + irradiance_measurement_id
    2. Auto-generation mode: Specify camera geometry, backend creates sensor + measurement
    """

    type: Literal["hcrf"] = "hcrf"
    id: Optional[str] = Field(None, description="Unique identifier")

    # Mode 1: Reference existing sensor + measurement
    radiance_sensor_id: Optional[str] = Field(
        None,
        description="ID of camera sensor providing conical radiance (must have FOV)",
    )
    irradiance_measurement_id: Optional[str] = Field(
        None, description="ID of IrradianceConfig providing BOA irradiance"
    )

    # Mode 2: Auto-generation mode
    platform_type: Optional[Literal["ground", "uav"]] = Field(
        None, description="Platform type for auto-generated camera"
    )
    viewing: Optional[Union[LookAtViewing, AngularFromOriginViewing]] = Field(
        None, description="Camera viewing geometry (auto-generation)"
    )
    fov: Optional[float] = Field(
        None, gt=0.0, le=180.0, description="Camera field of view (auto-generation)"
    )
    film_resolution: Optional[Tuple[int, int]] = Field(
        None, description="Camera resolution (auto-generation)"
    )
    reference_height_offset_m: Optional[float] = Field(
        None,
        description="Height offset for reference irradiance disk (auto-generation)",
    )
    samples_per_pixel: int = Field(64, description="Samples per pixel")
    terrain_relative_height: bool = Field(False, description="Terrain-relative height")
    post_processing: Optional[HCRFPostProcessingConfig] = Field(
        None, description="Post-processing configuration"
    )

    @model_validator(mode="after")
    def validate_hcrf_mode(self):
        """Validate that either sensor references OR auto-generation specs are provided."""
        has_refs = (
            self.radiance_sensor_id is not None
            and self.irradiance_measurement_id is not None
        )
        has_autogen = self.viewing is not None and self.fov is not None

        if has_refs and has_autogen:
            raise ValueError(
                "Specify either references OR auto-generation mode, not both"
            )

        if not has_refs and not has_autogen:
            raise ValueError(
                "Must specify either references (radiance_sensor_id + irradiance_measurement_id) "
                "OR auto-generation mode (viewing + fov)"
            )

        # Validate auto-generation mode
        if has_autogen:
            if not self.platform_type:
                raise ValueError("Auto-generation mode requires 'platform_type'")
            if not self.film_resolution:
                raise ValueError("Auto-generation mode requires 'film_resolution'")
            if self.reference_height_offset_m is None:
                raise ValueError(
                    "Auto-generation mode requires 'reference_height_offset_m'"
                )

        return self


class HemisphericalMeasurementLocation(BaseModel):
    """Location specification for hemispheric measurements.

    Used when measuring at a point with hemispherical integration
    (e.g., BOA irradiance, polar HDRF). Specifies target location
    using either geographic (lat/lon) or scene (x/y/z) coordinates.

    This class provides the location where a hemispheric measurement
    is performed, typically for measurements that integrate over all
    directions (hemisphere) at a specific point in space.
    """

    # Target location (EITHER lat/lon OR x/y/z, not both)
    target_lat: Optional[float] = Field(
        None,
        ge=-90.0,
        le=90.0,
        description="Target latitude in decimal degrees (WGS84)",
    )
    target_lon: Optional[float] = Field(
        None,
        ge=-180.0,
        le=180.0,
        description="Target longitude in decimal degrees (WGS84)",
    )

    target_x: Optional[float] = Field(
        None,
        description="Target X coordinate in scene coordinate system (meters)",
    )
    target_y: Optional[float] = Field(
        None,
        description="Target Y coordinate in scene coordinate system (meters)",
    )
    target_z: Optional[float] = Field(
        None,
        description="Target Z coordinate in scene coordinate system (meters)",
    )

    # Measurement parameters
    samples_per_pixel: int = Field(
        64,
        ge=1,
        description="Number of samples per pixel for Monte Carlo integration",
    )
    srf: Optional[SRFType] = Field(
        None,
        description="Spectral response function",
    )

    # Height offset (for terrain-relative measurements)
    height_offset_m: float = Field(
        0.0,
        ge=0.0,
        description="Height offset above terrain surface (meters)",
    )
    terrain_relative_height: bool = Field(
        True,
        description="If True, z-coordinates are interpreted as offsets from terrain elevation",
    )

    @model_validator(mode="after")
    def validate_coordinates(self) -> "HemisphericalMeasurementLocation":
        """Validate that either geographic or Cartesian coordinates are provided.

        Ensures:
        - Either lat/lon OR x/y/z are fully specified (not both)
        - No partial coordinate specifications
        """
        geo_present = self.target_lat is not None and self.target_lon is not None
        cartesian_present = (
            self.target_x is not None
            and self.target_y is not None
            and self.target_z is not None
        )

        any_geo = self.target_lat is not None or self.target_lon is not None
        if any_geo and not geo_present:
            raise ValueError(
                "If using geographic coordinates, both 'target_lat' and 'target_lon' are required"
            )

        any_cartesian = (
            self.target_x is not None
            or self.target_y is not None
            or self.target_z is not None
        )
        if any_cartesian and not cartesian_present:
            raise ValueError(
                "If using Cartesian coordinates, 'target_x', 'target_y', and 'target_z' are all required"
            )

        if geo_present and cartesian_present:
            raise ValueError(
                "Cannot specify both geographic (lat/lon) and Cartesian (x/y/z) coordinates"
            )

        if not geo_present and not cartesian_present:
            raise ValueError(
                "Must specify either geographic (lat/lon) or Cartesian (x/y/z) coordinates"
            )

        # Geographic mode: must have terrain_relative_height=True
        # (geographic coords don't provide absolute elevation, only lat/lon)
        if geo_present and not self.terrain_relative_height:
            raise ValueError(
                "When using geographic coordinates (lat/lon), terrain_relative_height must be True. "
                "Geographic coordinates require terrain elevation query to determine absolute height."
            )

        # Cartesian mode with absolute coordinates: ensure target_z is meaningful
        if cartesian_present and not self.terrain_relative_height:
            # target_z will be used as absolute elevation
            # (No additional validation needed - existing checks ensure target_z is present)
            pass

        return self


class RadianceConfig(HemisphericalMeasurementLocation):
    """Configuration for simple spectral radiance measurement.

    Measures upward radiance at a specific location without normalization.
    """

    type: Literal["radiance"] = "radiance"
    id: Optional[str] = Field(
        None,
        description="Unique identifier",
    )


class BHRConfig(HemisphericalMeasurementLocation):
    """Configuration for Bi-Hemispherical Reflectance (BHR) measurement.

    BHR integrates reflectance over all viewing and illumination directions.
    """

    type: Literal["bhr"] = "bhr"
    id: Optional[str] = Field(
        None,
        description="Unique identifier",
    )
    integration_mode: Literal["full", "isotropic"] = Field(
        default="full",
        description="Integration mode: 'full' for complete integration, 'isotropic' for isotropic assumption",
    )


# Union type with discriminator for type-safe measurement configs
MeasurementConfig = Annotated[
    Union[
        IrradianceConfig,
        BRFConfig,
        HDRFConfig,  # Now supports multiple instruments (radiancemeter, perspective, hemispherical)
        HCRFConfig,  # Hemispherical-Conical Reflectance Factor (camera FOV)
        RadianceConfig,
        BHRConfig,
    ],
    Field(discriminator="type"),
]


class BaseSensor(BaseModel):
    """Base sensor configuration."""

    id: Optional[str] = Field(
        None, description="Unique sensor identifier (auto-generated if not provided)"
    )
    platform_type: PlatformType
    viewing: ViewingType
    srf: Optional[SRFType] = Field(None, description="Spectral response function")
    produces: List[Literal["radiance", "irradiance", "hdrf", "brf", "bhr"]] = Field(
        ["radiance"],
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


class SRFPostProcessingConfig(BaseModel):
    """Base configuration for Gaussian SRF post-processing.

    This provides a common pattern for instruments that need
    post-processing SRF application (e.g., HYPSTAR, other spectrometers).

    Instruments can specify either:
    - Wavelength-dependent FWHM (vnir + swir) for different spectral ranges
    - Single FWHM for all wavelengths

    Example:
        # Wavelength-dependent (HYPSTAR-like)
        config = SRFPostProcessingConfig(
            apply_srf=True,
            fwhm_vnir_nm=3.0,
            fwhm_swir_nm=10.0
        )

        # Single FWHM (simpler instruments)
        config = SRFPostProcessingConfig(
            apply_srf=True,
            fwhm_nm=5.0
        )
    """

    apply_srf: bool = Field(default=True, description="Apply Gaussian SRF convolution")

    # Option 1: Wavelength-dependent FWHM (like HYPSTAR)
    fwhm_vnir_nm: Optional[float] = Field(
        default=None, gt=0.0, description="FWHM for VNIR range (<1000 nm) in nanometers"
    )
    fwhm_swir_nm: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="FWHM for SWIR range (>=1000 nm) in nanometers",
    )

    # Option 2: Single FWHM for all wavelengths
    fwhm_nm: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Single FWHM for all wavelengths in nanometers",
    )

    spatial_averaging: bool = Field(
        default=True, description="Average over spatial dimensions (x_index, y_index)"
    )
    spatial_statistic: Literal["mean", "median"] = Field(
        default="mean", description="Statistic for spatial averaging"
    )

    @model_validator(mode="after")
    def validate_fwhm(self):
        """Ensure at least one FWHM is specified when SRF is enabled."""
        if not self.apply_srf:
            return self

        has_dual = self.fwhm_vnir_nm is not None and self.fwhm_swir_nm is not None
        has_single = self.fwhm_nm is not None

        if not (has_dual or has_single):
            raise ValueError(
                "Must specify either (fwhm_vnir_nm + fwhm_swir_nm) or fwhm_nm when apply_srf=True"
            )

        if has_dual and has_single:
            raise ValueError(
                "Specify either dual FWHM (vnir+swir) OR single FWHM, not both"
            )

        return self


class HypstarPostProcessingConfig(SRFPostProcessingConfig):
    """Post-processing configuration for HYPSTAR sensor.

    HYPSTAR is a real instrument with specific spectral characteristics.
    This configuration inherits from SRFPostProcessingConfig and provides
    HYPSTAR-specific defaults for SRF convolution.

    For validation against real HYPSTAR observations, specify real_reference_file
    to interpolate simulation results to HYPSTAR's exact wavelength grid.

    Default HYPSTAR characteristics:
    - VNIR FWHM: 3.0 nm (wavelength < 1000 nm)
    - SWIR FWHM: 10.0 nm (wavelength >= 1000 nm)
    - Spatial averaging: enabled

    Example:
        # Basic usage (simulation wavelengths)
        config = HypstarPostProcessingConfig()

        # Validation mode (interpolate to L2A wavelengths)
        config = HypstarPostProcessingConfig(
            real_reference_file="HYPERNETS_L_GHNA_L2A_REF_*.nc"
        )
    """

    # Override defaults for HYPSTAR instrument
    fwhm_vnir_nm: float = Field(
        default=3.0, gt=0.0, description="HYPSTAR VNIR FWHM (default 3.0 nm)"
    )
    fwhm_swir_nm: float = Field(
        default=10.0, gt=0.0, description="HYPSTAR SWIR FWHM (default 10.0 nm)"
    )

    # L2A reference for wavelength interpolation
    real_reference_file: Optional[str] = Field(
        default=None,
        description=(
            "Path to HYPSTAR L2A NetCDF file for wavelength grid reference. "
            "If provided, SRF processing will interpolate to the L2A wavelengths "
            "instead of simulation wavelengths. This is required for validation "
            "against real HYPSTAR observations to ensure wavelength grids match. "
            "Example: 'HYPERNETS_L_GHNA_L2A_REF_20220517T0743_*.nc'"
        ),
    )
    wavelength_variable: str = Field(
        default="wavelength",
        description="Variable name for wavelengths in L2A reference file",
    )


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
    hypstar_post_processing: Optional[HypstarPostProcessingConfig] = Field(
        None,
        description="HYPSTAR-specific post-processing (auto-enabled for HYPSTAR instrument)",
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

                # Auto-enable HYPSTAR post-processing
                if self.hypstar_post_processing is None:
                    self.hypstar_post_processing = HypstarPostProcessingConfig()

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
    measurements: List[MeasurementConfig] = Field(
        default_factory=list,
        description="Unified list of radiative quantity measurements (HDRF, BRF, irradiance, etc.)",
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
        if not self.sensors and not self.measurements:
            raise ValueError("At least one sensor or measurement must be specified")

        self._validate_measurement_references()
        self._validate_sensor_references()

        return self

    def _validate_measurement_references(self):
        """Validate cross-references between measurements.

        Ensures that:
        - Measurement IDs are unique
        - reference_irradiance_id references an existing measurement
        - The referenced measurement is an IrradianceConfig
        """
        measurement_ids = set()
        irradiance_ids = set()

        for measurement in self.measurements:
            m_id = getattr(measurement, "id", None)
            if m_id:
                if m_id in measurement_ids:
                    raise ValueError(f"Duplicate measurement ID: '{m_id}'")
                measurement_ids.add(m_id)

                if isinstance(measurement, IrradianceConfig):
                    irradiance_ids.add(m_id)

        errors = []
        for measurement in self.measurements:
            ref_id = getattr(measurement, "irradiance_measurement_id", None)
            if ref_id is not None:
                m_id = getattr(measurement, "id", "unknown")

                if ref_id not in measurement_ids:
                    errors.append(
                        f"Measurement '{m_id}' references unknown measurement "
                        f"'{ref_id}' as irradiance_measurement_id"
                    )
                elif ref_id not in irradiance_ids:
                    errors.append(
                        f"Measurement '{m_id}' references '{ref_id}' as irradiance, "
                        f"but it is not an IrradianceConfig"
                    )

        if errors:
            raise ValueError(
                "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            )

    def _validate_sensor_references(self):
        """Validate that measurements reference existing sensors and measurements."""
        sensor_ids = {s.id for s in self.sensors}
        measurement_ids = {
            getattr(m, "id", None) for m in self.measurements if getattr(m, "id", None)
        }

        errors = []
        for measurement in self.measurements:
            # Check HDRF references
            if isinstance(measurement, HDRFConfig):
                if measurement.radiance_sensor_id:
                    if measurement.radiance_sensor_id not in sensor_ids:
                        errors.append(
                            f"HDRF '{measurement.id}' references unknown sensor "
                            f"'{measurement.radiance_sensor_id}'"
                        )
                if measurement.irradiance_measurement_id:
                    if measurement.irradiance_measurement_id not in measurement_ids:
                        errors.append(
                            f"HDRF '{measurement.id}' references unknown measurement "
                            f"'{measurement.irradiance_measurement_id}'"
                        )
                    # Validate that referenced measurement is IrradianceConfig
                    ref_measurement = next(
                        (
                            m
                            for m in self.measurements
                            if getattr(m, "id", None)
                            == measurement.irradiance_measurement_id
                        ),
                        None,
                    )
                    if ref_measurement and not isinstance(
                        ref_measurement, IrradianceConfig
                    ):
                        errors.append(
                            f"HDRF '{measurement.id}' references '{measurement.irradiance_measurement_id}' "
                            f"as irradiance, but it is not an IrradianceConfig"
                        )

            # Check HCRF references
            if isinstance(measurement, HCRFConfig):
                if measurement.radiance_sensor_id:
                    if measurement.radiance_sensor_id not in sensor_ids:
                        errors.append(
                            f"HCRF '{measurement.id}' references unknown sensor "
                            f"'{measurement.radiance_sensor_id}'"
                        )
                    # Validate that referenced sensor has FOV (is a camera)
                    sensor = self.get_sensor(measurement.radiance_sensor_id)
                    if sensor and not hasattr(sensor, "fov"):
                        errors.append(
                            f"HCRF '{measurement.id}' references sensor "
                            f"'{measurement.radiance_sensor_id}' which is not a camera (no FOV)"
                        )
                if measurement.irradiance_measurement_id:
                    if measurement.irradiance_measurement_id not in measurement_ids:
                        errors.append(
                            f"HCRF '{measurement.id}' references unknown measurement "
                            f"'{measurement.irradiance_measurement_id}'"
                        )
                    # Validate it's IrradianceConfig
                    ref_measurement = next(
                        (
                            m
                            for m in self.measurements
                            if getattr(m, "id", None)
                            == measurement.irradiance_measurement_id
                        ),
                        None,
                    )
                    if ref_measurement and not isinstance(
                        ref_measurement, IrradianceConfig
                    ):
                        errors.append(
                            f"HCRF '{measurement.id}' references '{measurement.irradiance_measurement_id}' "
                            f"as irradiance, but it is not an IrradianceConfig"
                        )

            # Check BRF sensor references
            if isinstance(measurement, BRFConfig):
                if measurement.radiance_sensor_id:
                    if measurement.radiance_sensor_id not in sensor_ids:
                        errors.append(
                            f"BRF '{measurement.id}' references unknown sensor "
                            f"'{measurement.radiance_sensor_id}'"
                        )
                if measurement.reference_radiance_sensor_id:
                    if measurement.reference_radiance_sensor_id not in sensor_ids:
                        errors.append(
                            f"BRF '{measurement.id}' references unknown sensor "
                            f"'{measurement.reference_radiance_sensor_id}'"
                        )

        if errors:
            raise ValueError(
                "Sensor reference validation failed:\n  - " + "\n  - ".join(errors)
            )

    @model_validator(mode="after")
    def validate_hdrf_hcrf_requires_irradiance(self):
        """Validate that HDRF/HCRF measurements have corresponding irradiance measurements.

        In reference mode, HDRF/HCRF must reference an existing IrradianceConfig.
        In auto-generation mode, the backend will create the irradiance measurement automatically.
        """
        # Get all irradiance measurement IDs
        irradiance_ids = {
            m.id for m in self.measurements if isinstance(m, IrradianceConfig)
        }

        errors = []
        for measurement in self.measurements:
            if isinstance(measurement, (HDRFConfig, HCRFConfig)):
                # In reference mode, must reference existing irradiance measurement
                if measurement.irradiance_measurement_id:
                    if measurement.irradiance_measurement_id not in irradiance_ids:
                        errors.append(
                            f"{measurement.__class__.__name__} '{measurement.id}' references "
                            f"irradiance_measurement_id='{measurement.irradiance_measurement_id}' "
                            f"which doesn't exist. Available: {sorted(irradiance_ids) if irradiance_ids else 'none'}"
                        )
                # In auto-gen mode, backend will create irradiance measurement
                # (no validation needed here, as it happens during backend execution)

        if errors:
            raise ValueError(
                "HDRF/HCRF validation failed:\n  - " + "\n  - ".join(errors)
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
        self, measurement: str
    ) -> List[Union[SatelliteSensor, UAVSensor, GroundSensor]]:
        """Get all sensors for a specific measurement type."""
        matching_sensors = []
        for sensor in self.sensors:
            if hasattr(sensor, "produces") and measurement in sensor.produces:
                matching_sensors.append(sensor)
        return matching_sensors

    @property
    def output_quantities(self) -> List[str]:
        """Get output quantities based on sensors."""
        # In the new system, sensors specify what they produce in the 'produces' field
        quantities = []
        for sensor in self.sensors:
            if hasattr(sensor, "produces"):
                quantities.extend(sensor.produces)
            else:
                # Default to radiance for all sensors
                quantities.append("radiance")
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
