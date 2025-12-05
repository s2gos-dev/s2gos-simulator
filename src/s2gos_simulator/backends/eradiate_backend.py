import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr
from PIL import Image
from s2gos_utils.coordinates import CoordinateSystem
from s2gos_utils.io.paths import open_file
from s2gos_utils.scene import SceneDescription
from upath import UPath

from .base import SimulationBackend
from .eradiate_materials import EradiateMaterialAdapter
from ..config import (
    AngularFromOriginViewing,
    ConstantIllumination,
    DirectionalIllumination,
    GroundInstrumentType,
    GroundSensor,
    HCRFConfig,
    HDRFConfig,
    HemisphericalViewing,
    IrradianceConfig,
    LookAtViewing,
    PlatformType,
    SatelliteInstrument,
    SatellitePlatform,
    SatelliteSensor,
    SimulationConfig,
    UAVInstrumentType,
    UAVSensor,
)
from ..hdrf_processor import HDRFProcessor
from ..irradiance_processor import IrradianceProcessor

try:
    import eradiate
    import mitsuba as mi
    from eradiate.experiments import AtmosphereExperiment
    from eradiate.radprops import AbsorptionDatabase
    from eradiate.scenes.atmosphere import (
        ExponentialParticleDistribution,
        GaussianParticleDistribution,
        HeterogeneousAtmosphere,
        HomogeneousAtmosphere,
        MolecularAtmosphere,
        ParticleLayer,
        UniformParticleDistribution,
    )
    from eradiate.units import unit_registry as ureg
    from eradiate.xarray.interp import dataarray_to_rgb

    ERADIATE_AVAILABLE = True
except ImportError:
    ERADIATE_AVAILABLE = False

HDRF_RAY_OFFSET = 0.1


def sanitize_sensor_id(sensor_id: str) -> str:
    """Sanitize sensor ID for Eradiate kernel compatibility.

    Replaces dots with underscores to prevent Eradiate from interpreting
    them as nested dictionary path separators.

    Example:
        'hypstar_wl500.3nm' → 'hypstar_wl500_3nm'

    Args:
        sensor_id: Original sensor ID

    Returns:
        Sanitized sensor ID safe for Eradiate kernel
    """
    return sensor_id.replace(".", "_") if sensor_id else sensor_id


class EradiateBackend(SimulationBackend):
    """
    Enhanced Eradiate backend for the new configuration system.
    """

    def __init__(self, simulation_config: SimulationConfig):
        """Initialize the Eradiate backend with new configuration system."""
        super().__init__(simulation_config)

        eradiate_hints = simulation_config.backend_hints.get("eradiate", {})
        self._eradiate_mode = eradiate_hints.get("mode", "mono")

        if ERADIATE_AVAILABLE:
            try:
                eradiate.set_mode(self._eradiate_mode)
            except Exception as e:
                raise ValueError(
                    f"Failed to set Eradiate mode '{self._eradiate_mode}': {e}. "
                    f"Check if this mode is supported in your Eradiate installation."
                )

    def is_available(self) -> bool:
        """Check if Eradiate dependencies are available."""
        return ERADIATE_AVAILABLE

    def _get_material_ids_from_scene(
        self, scene_description: SceneDescription
    ) -> List[str]:
        """Get material IDs from SceneDescription metadata.

        Args:
            scene_description: SceneDescription object

        Returns:
            List of material IDs in texture index order
        """
        material_indices = scene_description.material_indices

        material_ids = []
        for texture_index in sorted(material_indices.keys(), key=int):
            material_name = material_indices[texture_index]
            material_ids.append(material_name)

        return material_ids

    @property
    def supported_platforms(self) -> List[str]:
        """Eradiate supports all platform types."""
        return ["satellite", "uav", "ground"]

    @property
    def supported_measurements(self) -> List[str]:
        """Measurement types supported by Eradiate."""
        return [
            "radiance",
            "brf",
            "hdrf",
            "bhr",
            "bhr_iso",
            "fapar",
            "flux_3d",
            "irradiance",
            "boa_irradiance",
            "dhp",
        ]

    def validate_configuration(self) -> List[str]:
        """Validate configuration for Eradiate backend."""
        errors = super().validate_configuration()

        if not self.is_available():
            errors.append(
                "Eradiate is not available. Install with: pip install eradiate[kernel]"
            )

        valid_modes = {
            "mono",
            "ckd",
            "mono_polarized",
            "ckd_polarized",
            "mono_single",
            "mono_polarized_single",
            "mono_double",
            "mono_polarized_double",
            "ckd_single",
            "ckd_polarized_single",
            "ckd_double",
            "ckd_polarized_double",
            "none",
        }
        if self._eradiate_mode not in valid_modes:
            errors.append(
                f"Invalid Eradiate mode '{self._eradiate_mode}'. Valid modes: {sorted(valid_modes)}"
            )

        for sensor in self.simulation_config.sensors:
            if sensor.platform_type == PlatformType.SATELLITE:
                validation_error = self._validate_satellite_sensor(sensor)
                if validation_error:
                    errors.append(validation_error)
            elif sensor.platform_type == PlatformType.GROUND:
                if sensor.instrument not in [
                    GroundInstrumentType.HYPSTAR,
                    GroundInstrumentType.PERSPECTIVE_CAMERA,
                    GroundInstrumentType.PYRANOMETER,
                    GroundInstrumentType.FLUX_METER,
                    GroundInstrumentType.DHP_CAMERA,
                    GroundInstrumentType.RADIANCEMETER,
                ]:
                    errors.append(
                        f"Ground sensor {sensor.id} instrument type {sensor.instrument} is not supported"
                    )

        for measurement in self.simulation_config.measurements:
            if measurement.type not in self.supported_measurements:
                errors.append(
                    f"Measurement type '{measurement.type}' is not supported by Eradiate backend"
                )

        return errors

    def _validate_satellite_sensor(self, sensor) -> Optional[str]:
        """Validate satellite sensor platform/instrument/band combination using enum system."""
        from ..config import INSTRUMENT_BANDS, PLATFORM_INSTRUMENTS

        if sensor.platform == SatellitePlatform.CUSTOM:
            if sensor.srf is None:
                return "Custom platform requires explicit SRF configuration"
            return None

        try:
            platform_enum = SatellitePlatform(sensor.platform)
            instrument_enum = SatelliteInstrument(sensor.instrument)
        except ValueError as e:
            return f"Invalid platform or instrument: {e}"

        valid_instruments = PLATFORM_INSTRUMENTS.get(platform_enum, [])
        if instrument_enum not in valid_instruments:
            return f"Platform '{sensor.platform}' does not support instrument '{sensor.instrument}'. Valid instruments: {[inst.value for inst in valid_instruments]}"

        band_enum_class = INSTRUMENT_BANDS.get(instrument_enum)
        if band_enum_class is not None:
            try:
                band_enum_class(sensor.band)
            except ValueError:
                valid_bands = [band.value for band in band_enum_class]
                return f"Instrument '{sensor.platform}/{sensor.instrument}' does not support band '{sensor.band}'. Valid bands: {valid_bands}"

        return None

    def run_simulation(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: Optional[UPath] = None,
        **kwargs,
    ) -> xr.Dataset:
        """Run Eradiate simulation from scene description.

        Automatically detects HDRF measurements and executes dual simulation workflow
        (actual + white reference) if needed. For standard measurements, runs single
        simulation.

        Args:
            scene_description: Scene description from s2gos_generator
            scene_dir: Directory containing scene assets
            output_dir: Output directory (defaults to scene_dir/eradiate_renders)
            **kwargs: Additional options (plot_image, id_to_plot, etc.)

        Returns:
            xarray Dataset containing all simulation results (including HDRF if requested)
        """
        if not self.is_available():
            raise RuntimeError("Eradiate is not available")

        if output_dir is None:
            output_dir = scene_dir / "eradiate_renders"

        output_dir = UPath(output_dir)
        from s2gos_utils.io.paths import mkdir

        mkdir(output_dir)

        print(f"Eradiate simulation: {self.simulation_config.name}")
        print(f"  Sensors: {len(self.simulation_config.sensors)}")
        print(f"  Measurements: {len(self.simulation_config.measurements)}\n")

        hdrf_proc = HDRFProcessor(self)
        irr_proc = IrradianceProcessor(self)
        requires_hdrf = hdrf_proc.requires_hdrf()
        requires_irr = irr_proc.requires_irradiance()

        if requires_hdrf or requires_irr:
            print("Workflow: Standard + BOA Irradiance")
            return self._run_combined_workflow(
                scene_description, scene_dir, output_dir, irr_proc, **kwargs
            )
        else:
            print("Workflow: Standard")
            return self._run_standard_simulation(
                scene_description, scene_dir, output_dir, **kwargs
            )

    def _query_terrain_elevation(
        self, scene_description: SceneDescription, scene_dir: UPath, x: float, y: float
    ) -> float:
        """Query terrain elevation at scene coordinates (x, y) in meters.

        Args:
            scene_description: Scene description containing metadata
            scene_dir: Scene directory containing DEM data
            x: Scene x-coordinate in meters
            y: Scene y-coordinate in meters

        Returns:
            Elevation in meters at (x, y)

        Raises:
            FileNotFoundError: If DEM file not found
        """
        from s2gos_simulator.terrain_query import TerrainQuery

        terrain_query = TerrainQuery(scene_description, scene_dir)
        return terrain_query.query_elevation_at_scene_coords(x, y, raise_on_error=True)

    def _adjust_origin_target_for_terrain(
        self,
        viewing,
        scene_description: SceneDescription,
        scene_dir: UPath,
    ) -> tuple[list[float], Optional[list[float]]]:
        """Adjust origin and optionally target for terrain-relative positioning.

        Checks viewing geometry's terrain_relative_height flag and adjusts
        z-coordinates relative to terrain elevation from DEM.

        Args:
            viewing: Viewing geometry (LookAtViewing, AngularFromOriginViewing, etc.)
            scene_description: Scene description for terrain queries
            scene_dir: Scene directory for DEM access

        Returns:
            Tuple of (adjusted_origin, adjusted_target_or_None)
            - adjusted_origin: Origin with z adjusted for terrain elevation if flag set
            - adjusted_target: Target adjusted for terrain if LookAtViewing with flag set, else None
        """
        from ..config import LookAtViewing

        if not getattr(viewing, "terrain_relative_height", False):
            origin = list(viewing.origin)
            target = list(viewing.target) if hasattr(viewing, "target") else None
            return origin, target

        origin = list(viewing.origin)
        x, y, z_offset = origin
        terrain_elevation = self._query_terrain_elevation(
            scene_description, scene_dir, x, y
        )
        absolute_z = terrain_elevation + z_offset
        origin[2] = absolute_z

        logging.debug(
            f"Viewing origin: terrain={terrain_elevation:.2f}m, "
            f"offset={z_offset:.2f}m, final_z={absolute_z:.2f}m"
        )

        target = None
        if isinstance(viewing, LookAtViewing):
            target = list(viewing.target)
            target_x, target_y, target_z_offset = target
            target_terrain_elevation = self._query_terrain_elevation(
                scene_description, scene_dir, target_x, target_y
            )
            target_absolute_z = target_terrain_elevation + target_z_offset
            target[2] = target_absolute_z

            logging.debug(
                f"Viewing target: terrain={target_terrain_elevation:.2f}m, "
                f"offset={target_z_offset:.2f}m, final_z={target_absolute_z:.2f}m"
            )

        return origin, target

    def _run_standard_simulation(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: UPath,
        include_irradiance_measures: bool = True,
        **kwargs,
    ) -> xr.Dataset:
        """Run standard simulation (BRF, radiance, etc.)."""
        experiment = self._create_experiment(
            scene_description, scene_dir, include_irradiance_measures
        )

        for i in range(len(experiment.measures)):
            measure_id = getattr(experiment.measures[i], "id", f"measure_{i}")
            print(f"  Measure {i + 1}/{len(experiment.measures)}: {measure_id}")
            eradiate.run(experiment, measures=i)

        if kwargs.get("plot_image", False):
            self._create_rgb_visualization(
                experiment, output_dir, kwargs.get("id_to_plot", "rgb_camera")
            )

        return self._process_results(experiment, output_dir)

    def _run_combined_workflow(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: UPath,
        irradiance_processor: IrradianceProcessor,
        **kwargs,
    ) -> xr.Dataset:
        """Execute combined workflow with dual simulation for HDRF/HCRF measurements.

        This workflow is used when HDRF/HCRF measurements require BOA irradiance.
        IrradianceConfig modifies the scene by adding a white Lambertian disk,
        so it must be run separately to avoid affecting sensor measurements.

        Workflow Steps:
        1. Run actual scene simulation → sensor radiance results (L_raw)
        2. Apply post-processing → SRF convolution, spatial averaging (L_processed)
        3. Run white disk simulations → irradiance results (E_boa via reference disk)
        4. Compute derived measurements → HDRF = π×L_processed / E, HCRF = π×L_processed / E

        Key Architectural Decisions:
        - HDRF/HCRF ALWAYS require dual simulation (actual + reference disk)
        - Post-processing (SRF) MUST happen before computing HDRF/HCRF
        - Each IrradianceConfig needs separate simulation (scene modifications)
        - The π factor in HDRF/HCRF accounts for Lambertian white disk (ρ=1.0)

        Args:
            scene_description: Scene description from generator
            scene_dir: Scene directory path
            output_dir: Output directory for renders
            irradiance_processor: IrradianceProcessor instance for BOA measurements
            **kwargs: Additional options (plot_image, id_to_plot, etc.)

        Returns:
            xr.Dataset with combined results (sensors + irradiance + derived HDRF/HCRF)
        """
        print("\n[1/3] Radiance simulation (actual scene)...")
        actual_results = self._run_standard_simulation(
            scene_description,
            scene_dir,
            output_dir / "radiance",
            include_irradiance_measures=False,
            **kwargs,
        )

        print("\n[1.5/3] Applying post-processing to sensor results...")
        actual_dict = (
            actual_results
            if isinstance(actual_results, dict)
            else {"results": actual_results}
        )
        actual_dict = self._apply_post_processing_to_sensors(actual_dict)

        print("\n[2/3] BOA irradiance measurements (white disk)...")
        irr_results = irradiance_processor.execute_irradiance_measurements(
            scene_description, scene_dir, output_dir / "boa_irradiance"
        )

        combined_results = {**actual_dict, **irr_results}

        derived_results = self._compute_derived_measurements(combined_results)
        derived_output_dir = output_dir / "derived_results"
        for derived_name, dataset in derived_results.items():
            derived_output = (
                derived_output_dir
                / f"{self.simulation_config.name}_{derived_name}.zarr"
            )

            dataset.attrs["id"] = derived_name

            dataset.to_zarr(derived_output, mode="w")
            print(f"Derived measure '{derived_name}' saved to {derived_output}")

        postprocessed_results = {**combined_results, **derived_results}
        print(postprocessed_results.keys())
        print(
            f"\n✓ Combined workflow complete: {len(postprocessed_results)} total datasets"
        )
        return postprocessed_results

    def _create_hdrf_visualizations(
        self, hdrf_results: Dict[str, xr.Dataset], output_dir: UPath
    ) -> None:
        """Create visualizations for HDRF results.

        Args:
            hdrf_results: Dictionary of HDRF datasets
            output_dir: Output directory
        """
        import matplotlib.pyplot as plt

        vis_dir = output_dir / "hdrf_visualizations"
        from s2gos_utils.io.paths import mkdir

        mkdir(vis_dir)

        for measure_id, dataset in hdrf_results.items():
            try:
                hdrf_data = dataset["hdrf"]

                if "x_index" in hdrf_data.dims and "y_index" in hdrf_data.dims:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                    if "w" in hdrf_data.dims and len(hdrf_data.w) > 1:
                        img_data = hdrf_data.isel(w=0).values
                    else:
                        img_data = hdrf_data.squeeze().values

                    im = axes[0].imshow(img_data, cmap="RdYlGn", vmin=0, vmax=1)
                    axes[0].set_title(f"HDRF - {measure_id}")
                    axes[0].set_xlabel("X index")
                    axes[0].set_ylabel("Y index")
                    plt.colorbar(im, ax=axes[0], label="HDRF")

                    axes[1].hist(img_data.flatten(), bins=50, edgecolor="black")
                    axes[1].set_xlabel("HDRF")
                    axes[1].set_ylabel("Frequency")
                    axes[1].set_title("HDRF Distribution")
                    axes[1].axvline(
                        x=img_data.mean(), color="r", linestyle="--", label="Mean"
                    )
                    axes[1].legend()

                    plt.tight_layout()
                    output_file = vis_dir / f"{measure_id}_hdrf_visualization.png"
                    plt.savefig(output_file, dpi=150, bbox_inches="tight")
                    plt.close()

                    print(f"  Saved HDRF visualization: {output_file.name}")

                elif "w" in hdrf_data.dims:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    wavelengths = hdrf_data.w.values
                    hdrf_values = hdrf_data.values

                    ax.plot(wavelengths, hdrf_values, "b-", linewidth=2)
                    ax.set_xlabel("Wavelength (nm)")
                    ax.set_ylabel("HDRF")
                    ax.set_title(f"Spectral HDRF - {measure_id}")
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim([0, 1])

                    plt.tight_layout()
                    output_file = vis_dir / f"{measure_id}_hdrf_spectrum.png"
                    plt.savefig(output_file, dpi=150, bbox_inches="tight")
                    plt.close()

                    print(f"  Saved HDRF spectrum: {output_file.name}")

            except Exception as e:
                print(
                    f"  Warning: Could not create visualization for {measure_id}: {e}"
                )

    def _process_object_materials(self, scene_description: SceneDescription) -> None:
        """Validate that all object materials use string references only.

        Args:
            scene_description: Scene description with objects using string material references

        Raises:
            ValueError: If any object has invalid material reference
        """
        if not scene_description.objects:
            return

        for i, obj in enumerate(scene_description.objects):
            if "material" not in obj:
                continue

            material = obj["material"]

            if not isinstance(material, str):
                raise ValueError(
                    f"Object {i} has invalid material type {type(material).__name__}. "
                    "Only string references are supported. "
                    "Define materials in the scene's material library."
                )

            if material not in scene_description.materials:
                available = list(scene_description.materials.keys())
                raise ValueError(
                    f"Object {i} references unknown material '{material}'. "
                    f"Available materials: {available}"
                )

    def _expand_vegetation_collection(
        self, vegetation_collection_obj: dict, scene_dir: UPath, kdict: dict
    ):
        """Expand vegetation collection to individual Mitsuba instances efficiently.

        Args:
            vegetation_collection_obj: Vegetation collection object from scene description
            scene_dir: Scene directory for resolving paths
            kdict: Eradiate kernel dictionary to add instances to
        """
        data_file = vegetation_collection_obj["data_file"]
        binary_path = scene_dir / data_file

        try:
            vegetation_data = np.load(binary_path)
            count = len(vegetation_data)
            shapegroup_ref = vegetation_collection_obj["shapegroup_ref"]
            collection_name = vegetation_collection_obj.get(
                "collection_name", "vegetation"
            )

            logging.info(
                f"Expanding vegetation collection '{collection_name}' with {count} instances"
            )

            for i in range(count):
                instance_id = f"vegetation_instance_{collection_name}_{i}"

                x, y, z = (
                    float(vegetation_data[i]["x"]),
                    float(vegetation_data[i]["y"]),
                    float(vegetation_data[i]["z"]),
                )
                rotation = float(vegetation_data[i]["rotation"])
                scale = float(vegetation_data[i]["scale"])
                tilt_x = (
                    float(vegetation_data[i]["tilt_x"])
                    if "tilt_x" in vegetation_data.dtype.names
                    else 0.0
                )
                tilt_y = (
                    float(vegetation_data[i]["tilt_y"])
                    if "tilt_y" in vegetation_data.dtype.names
                    else 0.0
                )

                to_world = mi.ScalarTransform4f.translate([x, y, z])
                to_world = to_world @ mi.ScalarTransform4f.rotate([1, 0, 0], 90)
                to_world = to_world @ mi.ScalarTransform4f.rotate([0, 1, 0], rotation)
                to_world = to_world @ mi.ScalarTransform4f.rotate([1, 0, 0], tilt_x)
                to_world = to_world @ mi.ScalarTransform4f.rotate([0, 1, 0], tilt_y)
                to_world = to_world @ mi.ScalarTransform4f.scale(scale)

                kdict[instance_id] = {
                    "type": "instance",
                    "shapegroup": {"type": "ref", "id": shapegroup_ref},
                    "to_world": to_world,
                }

        except Exception as e:
            logging.error(
                f"Failed to expand vegetation collection '{vegetation_collection_obj.get('collection_name', 'unknown')}': {e}"
            )
            raise

    def _translate_materials(
        self, scene_description: SceneDescription, scene_dir: UPath
    ) -> tuple[dict, dict]:
        """Translate scene materials to Eradiate kernel dictionaries.

        Args:
            scene_description: Scene description with materials to translate
            scene_dir: Base directory for resolving file paths

        Returns:
            Tuple of (kdict, kpmap) containing material definitions
        """
        kdict = {}
        kpmap = {}
        adapter = EradiateMaterialAdapter()

        from s2gos_utils.scene.materials import (
            BilambertianMaterial,
            ConductorMaterial,
            DielectricMaterial,
            DiffuseMaterial,
            MeasuredMaterial,
            OceanLegacyMaterial,
            PlasticMaterial,
            PrincipledMaterial,
            RoughConductorMaterial,
            RPVMaterial,
        )

        for mat_name, material in scene_description.materials.items():
            if isinstance(material, DiffuseMaterial):
                mat_def = adapter.create_diffuse_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = adapter.create_diffuse_kpmap(material)
            elif isinstance(material, BilambertianMaterial):
                mat_def = adapter.create_bilambertian_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = adapter.create_bilambertian_kpmap(material)
            elif isinstance(material, RPVMaterial):
                mat_def = adapter.create_rpv_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = adapter.create_rpv_kpmap(material)
            elif isinstance(material, OceanLegacyMaterial):
                mat_def = adapter.create_ocean_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = adapter.create_ocean_kpmap(material)
            elif isinstance(material, DielectricMaterial):
                mat_def = adapter.create_dielectric_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = adapter.create_dielectric_kpmap(material)
            elif isinstance(material, ConductorMaterial):
                mat_def = adapter.create_conductor_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = adapter.create_conductor_kpmap(material)
            elif isinstance(material, RoughConductorMaterial):
                mat_def = adapter.create_rough_conductor_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = adapter.create_rough_conductor_kpmap(material)
            elif isinstance(material, PlasticMaterial):
                mat_def = adapter.create_plastic_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = adapter.create_plastic_kpmap(material)
            elif isinstance(material, PrincipledMaterial):
                mat_def = adapter.create_principled_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = adapter.create_principled_kpmap(material)
            elif isinstance(material, MeasuredMaterial):
                mat_def = adapter.create_measured_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = adapter.create_measured_kpmap(material)
            else:
                mat_def = adapter.create_diffuse_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = adapter.create_diffuse_kpmap(material)

            kdict.update(mat_kdict)
            kpmap.update(mat_kpmap)

        return kdict, kpmap

    def _add_hamster_materials(
        self,
        kdict: dict,
        kpmap: dict,
        scene_description: SceneDescription,
        hamster_data_dict: dict | None,
    ) -> None:
        """Add HAMSTER spatial albedo materials to kernel dictionaries.

        Modifies kdict and kpmap in place to add HAMSTER-based materials
        for each surface (target, buffer, background).

        Args:
            kdict: Kernel dictionary to update
            kpmap: Kernel parameter map to update
            scene_description: Scene description with materials
            hamster_data_dict: HAMSTER albedo data (None if not available)
        """
        if hamster_data_dict is None:
            return

        adapter = EradiateMaterialAdapter()

        region_material_names = {
            mat_name
            for idx, mat_name in scene_description.material_indices.items()
            if int(idx) >= 11
        }

        for surface_name, hamster_data in hamster_data_dict.items():
            for mat_name in scene_description.materials.keys():
                if mat_name in region_material_names:
                    continue

                hamster_material_id = f"_mat_{mat_name}_{surface_name}"

                hamster_kdict = adapter.create_hamster_kdict(
                    material_id=hamster_material_id, albedo_data=hamster_data
                )

                hamster_kpmap = adapter.create_hamster_kpmap(
                    material_id=hamster_material_id, albedo_data=hamster_data
                )

                kdict.update(hamster_kdict)
                kpmap.update(hamster_kpmap)

    def _create_transform_from_spec(self, transform_spec: dict):
        """Create Mitsuba transform from transform specification.

        Args:
            transform_spec: Dict with 'translate', 'rotate', 'scale' keys

        Returns:
            Mitsuba ScalarTransform4f
        """
        import mitsuba as mi

        to_world = mi.ScalarTransform4f()

        if "translate" in transform_spec:
            x, y, z = transform_spec["translate"]
            to_world = to_world @ mi.ScalarTransform4f.translate([x, y, z])

        if "rotate" in transform_spec:
            rx, ry, rz = transform_spec["rotate"]
            if rx != 0:
                to_world = to_world @ mi.ScalarTransform4f.rotate([1, 0, 0], rx)
            if ry != 0:
                to_world = to_world @ mi.ScalarTransform4f.rotate([0, 1, 0], ry)
            if rz != 0:
                to_world = to_world @ mi.ScalarTransform4f.rotate([0, 0, 1], rz)

        if "scale" in transform_spec:
            scale = transform_spec["scale"]
            to_world = to_world @ mi.ScalarTransform4f.scale(scale)

        return to_world

    def _create_transform_from_object(self, obj: dict):
        """Create Mitsuba transform from object position, rotation, scale.

        Args:
            obj: Object dict with 'position', 'rotation', 'scale' keys

        Returns:
            Mitsuba ScalarTransform4f
        """
        import mitsuba as mi

        x, y, z = obj["position"]
        scale = obj["scale"]

        to_world = mi.ScalarTransform4f.translate([x, y, z])

        if "rotation" in obj:
            rx, ry, rz = obj["rotation"]
            if rx != 0:
                to_world = to_world @ mi.ScalarTransform4f.rotate([1, 0, 0], rx)
            if ry != 0:
                to_world = to_world @ mi.ScalarTransform4f.rotate([0, 1, 0], ry)
            if rz != 0:
                to_world = to_world @ mi.ScalarTransform4f.rotate([0, 0, 1], rz)

        to_world = to_world @ mi.ScalarTransform4f.scale(scale)
        return to_world

    def _create_ply_shape_dict(self, value: dict, scene_dir: UPath) -> dict:
        """Create PLY shape dictionary for use in shapegroups.

        Args:
            value: PLY shape specification
            scene_dir: Base directory for mesh paths

        Returns:
            Mitsuba shape dictionary
        """
        shape_dict = {"type": "ply"}

        if "filename" in value:
            mesh_path = scene_dir / value["filename"]
            shape_dict["filename"] = str(mesh_path)

        if "face_normals" in value:
            shape_dict["face_normals"] = value["face_normals"]

        if "bsdf" in value:
            shape_dict["bsdf"] = value["bsdf"]

        return shape_dict

    def _add_shapegroup(self, kdict: dict, obj: dict, scene_dir: UPath) -> None:
        """Add a shapegroup object to kernel dictionary.

        Shapegroups are collections of shapes that can be instanced.

        Args:
            kdict: Kernel dictionary to update
            obj: Shapegroup object specification
            scene_dir: Base directory for resolving mesh paths
        """
        obj_dict = {"type": "shapegroup"}
        if "id" in obj:
            obj_dict["id"] = obj["id"]

        for key, value in obj.items():
            if key in ["type", "id", "object_id"]:
                continue

            if isinstance(value, dict) and value.get("type") == "ply":
                shape_dict = self._create_ply_shape_dict(value, scene_dir)
                obj_dict[key] = shape_dict
            elif isinstance(value, dict) and value.get("type") in [
                "sphere",
                "cube",
                "cylinder",
                "rectangle",
                "disk",
            ]:
                obj_dict[key] = value
            elif not isinstance(value, dict):
                obj_dict[key] = value
            else:
                logging.warning(
                    f"Skipping unrecognized entry in shapegroup '{key}': "
                    f"{value.get('type', 'unknown')}"
                )

        obj_id = obj.get("id") or obj.get("object_id", f"shapegroup_{len(kdict)}")
        kdict[obj_id] = obj_dict

    def _add_instance(self, kdict: dict, obj: dict) -> None:
        """Add an instance object to kernel dictionary.

        Instances reference shapegroups with transformations.

        Args:
            kdict: Kernel dictionary to update
            obj: Instance object specification
        """
        obj_dict = {"type": "instance", "shapegroup": obj["shapegroup"]}
        if "object_id" in obj:
            obj_dict["id"] = obj["object_id"]

        if "to_world" in obj:
            transform_spec = obj["to_world"]
            if transform_spec.get("type") == "transform":
                to_world = self._create_transform_from_spec(transform_spec)
                obj_dict["to_world"] = to_world
            else:
                obj_dict["to_world"] = obj["to_world"]

        obj_id = obj.get("id") or obj.get("object_id", f"instance_{len(kdict)}")
        kdict[obj_id] = obj_dict

    def _add_disk(self, kdict: dict, obj: dict) -> None:
        """Add a disk object to kernel dictionary.

        Creates a circular disk with Lambertian white material.

        Args:
            kdict: Kernel dictionary to update
            obj: Disk object specification
        """
        import mitsuba as mi

        center = obj["center"]
        radius = obj["radius"]
        disk_id = obj.get("id") or obj.get("object_id", f"disk_{len(kdict)}")

        obj_dict = {
            "type": "disk",
            "to_world": (
                mi.ScalarTransform4f.translate(center)
                @ mi.ScalarTransform4f.scale(radius)
            ),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "uniform", "value": 1.0},
            },
            "id": disk_id,
        }

        kdict[disk_id] = obj_dict

    def _add_ply_mesh(self, kdict: dict, obj: dict, scene_dir: UPath) -> None:
        """Add a PLY mesh object to kernel dictionary.

        Handles mesh loading with optional materials, transforms, and face normals.

        Args:
            kdict: Kernel dictionary to update
            obj: PLY mesh object specification
            scene_dir: Base directory for resolving mesh paths
        """
        object_mesh_path = scene_dir / obj["mesh"]
        obj_dict = {
            "type": "ply",
            "filename": str(object_mesh_path),
            "id": obj["id"],
        }

        if "face_normals" in obj:
            obj_dict["face_normals"] = obj["face_normals"]

        if "material" in obj:
            material = obj["material"]
            if isinstance(material, str):
                obj_dict["bsdf"] = {"type": "ref", "id": f"_mat_{material}"}
            else:
                obj_dict["bsdf"] = {
                    "type": "diffuse",
                    "reflectance": {"type": "uniform", "value": 0.5},
                }

        if "position" in obj and "scale" in obj:
            to_world = self._create_transform_from_object(obj)
            obj_dict["to_world"] = to_world

        obj_id = obj.get("id") or obj.get("object_id", f"object_{len(kdict)}")
        kdict[obj_id] = obj_dict

    def _add_scene_objects(
        self, kdict: dict, scene_description: SceneDescription, scene_dir: UPath
    ) -> None:
        """Add 3D objects from scene description to kernel dictionary.

        Processes all object types: shapegroups, instances, vegetation
        collections, disks, and PLY meshes.

        Args:
            kdict: Kernel dictionary to update
            scene_description: Scene with objects to process
            scene_dir: Base directory for resolving mesh paths
        """
        if not scene_description.objects:
            return

        logging.info(f"Processing {len(scene_description.objects)} objects")

        for obj in scene_description.objects:
            obj_type = obj.get("type", "ply")

            if obj_type == "shapegroup":
                self._add_shapegroup(kdict, obj, scene_dir)
            elif obj_type == "instance":
                self._add_instance(kdict, obj)
            elif obj_type == "vegetation_collection":
                self._expand_vegetation_collection(obj, scene_dir, kdict)
            elif obj_type == "disk":
                self._add_disk(kdict, obj)
            else:
                self._add_ply_mesh(kdict, obj, scene_dir)

    def _create_experiment(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        include_irradiance_measures: bool = True,
    ):
        """Create Eradiate experiment from scene description.

        Args:
            scene_description: Scene description with all scene data
            scene_dir: Base directory for resolving file paths
            include_irradiance_measures: Whether to include irradiance measurements

        Returns:
            AtmosphereExperiment configured for the scene
        """
        self._current_scene_dir = scene_dir
        self._current_scene_description = scene_description

        generated_sensor_ids = self._auto_generate_sensors_for_measurements()
        if generated_sensor_ids:
            print(f"  Auto-generated {len(generated_sensor_ids)} sensors:")
            for sensor_id in generated_sensor_ids:
                print(f"    - {sensor_id}")

        self._process_object_materials(scene_description)

        kdict, kpmap = self._translate_materials(scene_description, scene_dir)

        hamster_data_dict = self._get_hamster_data_for_scene(
            scene_description, scene_dir
        )
        self._add_hamster_materials(kdict, kpmap, scene_description, hamster_data_dict)

        buffer = scene_description.buffer
        background = scene_description.background

        hamster_available = hamster_data_dict is not None
        kdict.update(
            self._create_target_surface(scene_description, scene_dir, hamster_available)
        )

        if buffer:
            kdict.update(
                self._create_buffer_surface(
                    scene_description, scene_dir, hamster_available
                )
            )

        if background:
            kdict.update(
                self._create_background_surface(
                    scene_description, scene_dir, hamster_available
                )
            )

        self._add_scene_objects(kdict, scene_description, scene_dir)

        atmosphere = self._create_atmosphere_from_config(scene_description)

        illumination = self._translate_illumination()

        measures = self._translate_sensors(
            scene_description, include_irradiance_measures
        )

        geometry = self._create_geometry_from_atmosphere(scene_description)

        self._validate_material_ids(kdict, scene_description)
        return AtmosphereExperiment(
            geometry=geometry,
            atmosphere=atmosphere,
            surface=None,
            illumination=illumination,
            measures=measures,
            kdict=kdict,
            kpmap=kpmap,
        )

    def _create_geometry_from_atmosphere(self, scene_description: SceneDescription):
        """Create geometry with bounds matching the atmosphere configuration."""
        atmosphere = scene_description.atmosphere
        toa = atmosphere["toa"] if "toa" in atmosphere else 40000.0

        geometry = {
            "type": "plane_parallel",
            "toa_altitude": toa,
        }

        return geometry

    def _create_atmosphere_from_config(self, scene_description: SceneDescription):
        """Create atmosphere based on scene description format."""
        atmosphere = scene_description.atmosphere
        atmosphere_type = atmosphere["type"] if "type" in atmosphere else None

        if not atmosphere_type:
            raise ValueError("Atmosphere configuration must specify 'type' field")

        if atmosphere_type == "molecular":
            return self._create_molecular_atmosphere_from_scene(atmosphere)
        elif atmosphere_type == "homogeneous":
            return self._create_homogeneous_atmosphere_from_scene(atmosphere)
        elif atmosphere_type == "heterogeneous":
            return self._create_heterogeneous_atmosphere_from_scene(atmosphere)
        else:
            raise ValueError(f"Unknown atmosphere type: {atmosphere_type}")

    def _create_molecular_atmosphere_from_dict(self, mol_dict):
        """Create molecular atmosphere from scene description.

        Supports either joseki identifiers or CAMS NetCDF files.
        """
        if "thermoprops_file" in mol_dict:
            import xarray as xr
            from upath import UPath

            thermoprops_file = UPath(mol_dict["thermoprops_file"])
            thermoprops = xr.open_dataset(thermoprops_file).squeeze(drop=True)
        else:
            thermoprops_id = mol_dict.get(
                "thermoprops_identifier", "afgl_1986-us_standard"
            )
            altitude_min = mol_dict.get("altitude_min", 0.0)
            altitude_max = mol_dict.get("altitude_max", 120000.0)
            num_steps = (
                (altitude_max - altitude_min) / mol_dict.get("altitude_step", 1000)
            ) + 1

            thermoprops = {
                "identifier": thermoprops_id,
                "z": np.linspace(altitude_min, altitude_max, int(num_steps)) * ureg.m,
            }

        absorption_data = (
            mol_dict.get("absorption_database") or AbsorptionDatabase.default()
        )

        atmosphere = MolecularAtmosphere(
            thermoprops=thermoprops,
            absorption_data=absorption_data,
            has_absorption=mol_dict.get("has_absorption", True),
            has_scattering=mol_dict.get("has_scattering", True),
        )

        return atmosphere

    def _create_particle_layer_from_dict(self, layer_dict):
        """Create particle layer directly from scene description data."""
        dist_type = layer_dict.get("distribution_type", "exponential")
        if dist_type == "exponential":
            if "rate" in layer_dict.keys():
                if "scale" in layer_dict.keys():
                    print(
                        "WARNING: scale and rate should be mutually exclusive in exponential distribution, using rate"
                    )
                distribution = ExponentialParticleDistribution(
                    scale=layer_dict.get("rate", 5.0)
                )
            else:
                distribution = ExponentialParticleDistribution(
                    rate=layer_dict.get("scale", 0.2)
                )
        elif dist_type == "gaussian":
            distribution = GaussianParticleDistribution(
                mean=layer_dict.get("center_altitude", 0.5),
                std=layer_dict.get("width", 1 / 6),
            )
        else:
            distribution = UniformParticleDistribution(
                {"bounds": layer_dict.get("bounds", [0, 1])}
            )

        layer = ParticleLayer(
            dataset=layer_dict["aerosol_dataset"],
            tau_ref=layer_dict["optical_thickness"],
            w_ref=layer_dict.get("reference_wavelength", 550.0),
            bottom=layer_dict["altitude_bottom"],
            top=layer_dict["altitude_top"],
            distribution=distribution,
            has_absorption=layer_dict.get("has_absorption", True),
        )

        return layer

    def _create_molecular_atmosphere_from_scene(self, atmosphere_dict):
        """Create molecular atmosphere from scene description."""
        if "molecular_atmosphere" in atmosphere_dict:
            mol_dict = atmosphere_dict["molecular_atmosphere"]
            return self._create_molecular_atmosphere_from_dict(mol_dict)
        else:
            return self._create_molecular_atmosphere_from_dict({})

    def _create_homogeneous_atmosphere_from_scene(self, atmosphere_dict):
        """Create homogeneous atmosphere from scene description."""
        atmosphere = HomogeneousAtmosphere(
            boa=atmosphere_dict.get("boa", 0.0),
            toa=atmosphere_dict.get("toa", 40000.0),
            particle_layers=[
                ParticleLayer(
                    dataset=atmosphere_dict.get("aerosol_ds", "sixsv-continental"),
                    optical_thickness=atmosphere_dict.get("aerosol_ot", 0.1),
                    altitude_bottom=atmosphere_dict.get("boa", 0.0),
                    altitude_top=atmosphere_dict.get("toa", 40000.0),
                    reference_wavelength=550.0,
                )
            ],
        )

        return atmosphere

    def _create_heterogeneous_atmosphere_from_scene(self, atmosphere_dict):
        """Create heterogeneous atmosphere from scene description."""
        has_molecular = (
            atmosphere_dict.get("has_molecular_atmosphere", False)
            or "molecular_atmosphere" in atmosphere_dict
        )
        has_particles = (
            atmosphere_dict.get("has_particle_layers", False)
            or "particle_layers" in atmosphere_dict
        )

        molecular_atmosphere = None
        particle_layers = []

        if has_molecular:
            mol_dict = atmosphere_dict["molecular_atmosphere"]
            molecular_atmosphere = self._create_molecular_atmosphere_from_dict(mol_dict)

        if has_particles:
            for layer_dict in atmosphere_dict["particle_layers"]:
                layer = self._create_particle_layer_from_dict(layer_dict)
                if layer:
                    particle_layers.append(layer)

        atmosphere = HeterogeneousAtmosphere(
            molecular_atmosphere=molecular_atmosphere, particle_layers=particle_layers
        )

        return atmosphere

    def _translate_illumination(self) -> Dict[str, Any]:
        """Translate generic illumination to Eradiate format."""
        illumination = self.simulation_config.illumination

        if isinstance(illumination, DirectionalIllumination):
            return {
                "type": "directional",
                "id": illumination.id,
                "zenith": illumination.zenith * ureg.deg,
                "azimuth": illumination.azimuth * ureg.deg,
                "irradiance": {
                    "type": "solar_irradiance",
                    "dataset": illumination.irradiance_dataset,
                },
            }
        elif isinstance(illumination, ConstantIllumination):
            return {
                "type": "constant",
                "id": illumination.id,
                "radiance": illumination.radiance,
            }
        else:
            raise ValueError(f"Unsupported illumination type: {type(illumination)}")

    def _calculate_target_from_angles(
        self, view: AngularFromOriginViewing
    ) -> tuple[list[float], list[float]]:
        """
        Calculates a target point and direction vector from an AngularFromOriginViewing object.

        Returns:
            A tuple containing (target_position, direction_vector).
        """
        zen_rad = np.deg2rad(view.zenith)
        az_rad = np.deg2rad(view.azimuth)

        direction = np.array(
            [
                np.sin(zen_rad) * np.cos(az_rad),
                np.sin(zen_rad) * np.sin(az_rad),
                np.cos(zen_rad),
            ]
        )

        origin_vec = np.array(view.origin)
        target_vec = origin_vec + direction * 1000.0

        return target_vec.tolist(), direction.tolist()

    def _viewing_angles_to_direction(
        self, zenith: float, azimuth: float
    ) -> list[float]:
        """Convert viewing angles to direction vector for distant measure.

        Args:
            zenith: Viewing zenith angle in degrees (0° = nadir, 90° = horizon)
            azimuth: Viewing azimuth angle in degrees (0° = North, 90° = East)

        Returns:
            Direction vector [x, y, z] pointing from distant sensor toward scene.
            Uses coordinate system: x = East, y = North, z = Up.
        """
        zen_rad = np.deg2rad(zenith)
        az_rad = np.deg2rad(azimuth)

        direction = [
            np.sin(zen_rad) * np.sin(az_rad),
            np.sin(zen_rad) * np.cos(az_rad),
            -np.cos(zen_rad),
        ]

        return direction

    def _resolve_viewing_geometry(
        self,
        config,
        viewing: Union[LookAtViewing, AngularFromOriginViewing],
        scene_description: SceneDescription,
        scene_dir: UPath,
    ) -> tuple[list[float], list[float]]:
        """Resolve viewing geometry to origin and target coordinates.

        Applies terrain-relative adjustment if viewing geometry has the flag set.

        Args:
            config: Measurement config (for future use)
            viewing: Viewing geometry specification
            scene_description: Scene description for terrain elevation queries
            scene_dir: Path to scene directory containing DEM

        Returns:
            Tuple of (origin, target) as lists of [x, y, z] coordinates
        """
        origin, target = self._adjust_origin_target_for_terrain(
            viewing, scene_description, scene_dir
        )

        if isinstance(viewing, AngularFromOriginViewing) and target is None:
            target, _ = self._calculate_target_from_angles(viewing)

        return origin, target

    def _translate_sensors(
        self,
        scene_description: SceneDescription,
        include_irradiance_measures: bool = True,
    ) -> List[Dict[str, Any]]:
        """Translate generic sensors and radiative quantities to Eradiate measures."""
        measures = []

        for sensor in self.simulation_config.sensors:
            if isinstance(sensor, SatelliteSensor):
                measures.append(
                    self._translate_satellite_sensor(sensor, scene_description)
                )
            elif isinstance(sensor, UAVSensor):
                measures.append(
                    self._translate_uav_sensor(
                        sensor,
                        self._current_scene_description,
                        self._current_scene_dir,
                    )
                )
            elif isinstance(sensor, GroundSensor):
                measures.append(
                    self._translate_ground_sensor(
                        sensor,
                        self._current_scene_description,
                        self._current_scene_dir,
                    )
                )
            else:
                raise ValueError(f"Unsupported sensor type: {type(sensor)}")

        from ..config import (
            HCRFConfig,
            HDRFConfig,
            IrradianceConfig,
        )

        for measurement in self.simulation_config.measurements:
            if isinstance(measurement, IrradianceConfig):
                measures.append(self._create_irradiance_measure(measurement))
            elif isinstance(measurement, HDRFConfig) or isinstance(
                measurement, HCRFConfig
            ):
                continue
            else:
                raise ValueError(f"Unsupported measurement type: {type(measurement)}")

        return measures

    def _translate_satellite_sensor(
        self, sensor: SatelliteSensor, scene_description: SceneDescription
    ) -> Dict[str, Any]:
        """Translate satellite sensor to Eradiate mpdistant measure using scene coordinate system."""

        scene_location = scene_description.location
        scene_center_lat = scene_location.get("center_lat")
        scene_center_lon = scene_location.get("center_lon")

        if scene_center_lat is None or scene_center_lon is None:
            raise ValueError(
                "Scene description missing center_lat or center_lon in location"
            )

        coords = CoordinateSystem(scene_center_lat, scene_center_lon)

        if isinstance(sensor.target_size_km, (int, float)):
            width_km = height_km = sensor.target_size_km
        else:
            width_km, height_km = sensor.target_size_km

        target_bounds = coords.create_rectangle(
            sensor.target_center_lat, sensor.target_center_lon, width_km, height_km
        )

        measure_config = {
            "type": "mpdistant",
            "construct": "from_angles",
            "angles": [sensor.viewing.zenith, sensor.viewing.azimuth],
            "id": sanitize_sensor_id(sensor.id),
            "film_resolution": sensor.film_resolution,
            "target": {
                "type": "rectangle",
                "xmin": target_bounds["xmin"],
                "xmax": target_bounds["xmax"],
                "ymin": target_bounds["ymin"],
                "ymax": target_bounds["ymax"],
            },
            "srf": self._translate_srf(sensor.srf),
            "spp": sensor.samples_per_pixel,
        }

        return measure_config

    def _translate_uav_sensor(
        self,
        sensor: UAVSensor,
        scene_description: SceneDescription,
        scene_dir: UPath,
    ) -> Dict[str, Any]:
        """Translate UAV sensor to Eradiate measure.

        Handles terrain-relative height positioning if enabled on viewing geometry.
        """
        view = sensor.viewing

        origin, target = self._adjust_origin_target_for_terrain(
            view, scene_description, scene_dir
        )

        base_config = {
            "id": sanitize_sensor_id(sensor.id),
            "spp": sensor.samples_per_pixel,
            "srf": self._translate_srf(sensor.srf),
            "origin": origin,
        }

        if sensor.instrument == UAVInstrumentType.PERSPECTIVE_CAMERA:
            base_config["type"] = "perspective"
            base_config["fov"] = sensor.fov or 70.0
            base_config["film_resolution"] = sensor.resolution or [1024, 1024]

            if isinstance(view, LookAtViewing):
                base_config["target"] = target if target is not None else view.target
                base_config["up"] = view.up or [0, 0, 1]
            elif isinstance(view, AngularFromOriginViewing):
                target, _ = self._calculate_target_from_angles(view)
                base_config["target"] = target
                base_config["up"] = view.up or [0, 0, 1]

        elif sensor.instrument == UAVInstrumentType.RADIANCEMETER:
            base_config["type"] = "radiancemeter"

            if isinstance(view, LookAtViewing):
                base_config["target"] = target if target is not None else view.target
            elif isinstance(view, AngularFromOriginViewing):
                target, _ = self._calculate_target_from_angles(view)
                base_config["target"] = target

        else:
            raise ValueError(f"Unsupported UAV instrument type: {sensor.instrument}")

        return base_config

    def _translate_ground_sensor(
        self,
        sensor: GroundSensor,
        scene_description: SceneDescription,
        scene_dir: UPath,
    ) -> Dict[str, Any]:
        """Translate ground sensor to Eradiate measure.

        Handles terrain-relative height positioning if enabled on viewing geometry.
        For HYPSTAR sensors with post-processing enabled, uses wide uniform SRF
        during simulation; Gaussian SRF is applied in post-processing.
        """
        view = sensor.viewing

        if (
            sensor.instrument == GroundInstrumentType.HYPSTAR
            and sensor.hypstar_post_processing
            and sensor.hypstar_post_processing.apply_srf
        ):
            simulation_srf = {
                "type": "uniform",
                "wmin": 380.0,
                "wmax": 1680.0,
                "value": 1.0,
            }
        else:
            simulation_srf = self._translate_srf(sensor.srf)

        base_measure = {
            "id": sanitize_sensor_id(sensor.id),
            "spp": sensor.samples_per_pixel,
            "srf": simulation_srf,
        }

        if isinstance(view, HemisphericalViewing):
            base_measure["type"] = "hdistant"
            base_measure["direction"] = [0, 0, 1] if view.upward_looking else [0, 0, -1]

        elif isinstance(view, (LookAtViewing, AngularFromOriginViewing)):
            origin, target = self._adjust_origin_target_for_terrain(
                view, scene_description, scene_dir
            )

            base_measure["origin"] = origin

            if sensor.instrument in [
                GroundInstrumentType.PERSPECTIVE_CAMERA,
                GroundInstrumentType.DHP_CAMERA,
            ]:
                base_measure["type"] = "perspective"
                base_measure["film_resolution"] = sensor.resolution or [1024, 1024]
                base_measure["fov"] = sensor.fov or (
                    180.0
                    if sensor.instrument == GroundInstrumentType.DHP_CAMERA
                    else 70.0
                )
                base_measure["up"] = view.up or [0, 0, 1]

            elif sensor.instrument == GroundInstrumentType.HYPSTAR:
                base_measure["type"] = "perspective"
                base_measure["film_resolution"] = sensor.resolution or [5, 5]
                base_measure["fov"] = sensor.fov or 5.0
                base_measure["up"] = view.up or [0, 0, 1]

            elif sensor.instrument == GroundInstrumentType.RADIANCEMETER:
                base_measure["type"] = "radiancemeter"

            else:
                base_measure["type"] = "radiancemeter"

            if isinstance(view, LookAtViewing):
                base_measure["target"] = target if target is not None else view.target

            elif isinstance(view, AngularFromOriginViewing):
                target, direction = self._calculate_target_from_angles(view)
                base_measure["target"] = target

        else:
            raise ValueError(
                f"Unsupported viewing type for ground sensor: {type(view)}"
            )

        return base_measure

    def _auto_generate_sensors_for_measurements(self) -> List[str]:
        """Auto-generate sensors for measurements that require them.

        Returns:
            List of generated sensor IDs
        """
        generated_sensor_ids = []

        for measurement in self.simulation_config.measurements:
            if (
                isinstance(measurement, HDRFConfig)
                and measurement.radiance_sensor_id is None
            ):
                radiance_sensor = self._generate_radiance_sensor_for_hdrf(measurement)
                irradiance_measurement = self._generate_irradiance_measurement_for_hdrf(
                    measurement
                )

                self.simulation_config.sensors.append(radiance_sensor)
                self.simulation_config.measurements.append(irradiance_measurement)

                measurement.radiance_sensor_id = radiance_sensor.id
                measurement.irradiance_measurement_id = irradiance_measurement.id

                generated_sensor_ids.append(radiance_sensor.id)
                logging.info(
                    f"Auto-generated radiance sensor '{radiance_sensor.id}' and irradiance measurement '{irradiance_measurement.id}' for HDRF '{measurement.id}'"
                )

            elif (
                isinstance(measurement, HCRFConfig)
                and measurement.radiance_sensor_id is None
            ):
                camera_sensor = self._generate_camera_sensor_for_hcrf(measurement)
                irradiance_measurement = self._generate_irradiance_measurement_for_hcrf(
                    measurement
                )

                self.simulation_config.sensors.append(camera_sensor)
                self.simulation_config.measurements.append(irradiance_measurement)

                measurement.radiance_sensor_id = camera_sensor.id
                measurement.irradiance_measurement_id = irradiance_measurement.id

                generated_sensor_ids.append(camera_sensor.id)
                logging.info(
                    f"Auto-generated camera sensor '{camera_sensor.id}' and irradiance measurement '{irradiance_measurement.id}' for HCRF '{measurement.id}'"
                )

        return generated_sensor_ids

    def _generate_radiance_sensor_for_hdrf(
        self, hdrf_config: HDRFConfig
    ) -> GroundSensor:
        """Generate radiance sensor for HDRF measurement.

        Args:
            hdrf_config: HDRFConfig in auto-generation mode

        Returns:
            GroundSensor configured for radiance measurement
        """
        sensor_id = (
            f"{hdrf_config.id}_radiance" if hdrf_config.id else "hdrf_radiance_sensor"
        )

        if hdrf_config.instrument == "radiancemeter":
            return GroundSensor(
                id=sensor_id,
                instrument=GroundInstrumentType.RADIANCEMETER,
                viewing=hdrf_config.viewing,
                samples_per_pixel=hdrf_config.samples_per_pixel,
                srf=hdrf_config.srf,
            )
        elif hdrf_config.instrument == "hemispherical":
            if not hdrf_config.location:
                raise ValueError("Hemispherical HDRF requires 'location' field")

            if (
                hdrf_config.location.target_lat is not None
                and hdrf_config.location.target_lon is not None
            ):
                coords = CoordinateSystem(
                    self._current_scene_description.location["center_lat"],
                    self._current_scene_description.location["center_lon"],
                )
                origin_x, origin_y = coords.latlon_to_scene(
                    hdrf_config.location.target_lat, hdrf_config.location.target_lon
                )
                origin_z = (
                    hdrf_config.location.height_offset_m
                    if hdrf_config.location.terrain_relative_height
                    else hdrf_config.location.target_z
                )
            else:
                origin_x = hdrf_config.location.target_x
                origin_y = hdrf_config.location.target_y
                origin_z = (
                    hdrf_config.location.height_offset_m
                    if hdrf_config.location.terrain_relative_height
                    else hdrf_config.location.target_z
                )

            return GroundSensor(
                id=sensor_id,
                instrument=GroundInstrumentType.RADIANCEMETER,
                viewing=HemisphericalViewing(
                    origin=[origin_x, origin_y, origin_z],
                    upward_looking=False,
                    terrain_relative_height=hdrf_config.location.terrain_relative_height,
                ),
                samples_per_pixel=hdrf_config.samples_per_pixel,
                srf=hdrf_config.srf,
            )
        else:
            raise ValueError(
                f"Unsupported HDRF instrument type: {hdrf_config.instrument}"
            )

    def _generate_irradiance_measurement_for_hdrf(
        self, hdrf_config: HDRFConfig
    ) -> IrradianceConfig:
        """Generate irradiance measurement for HDRF.

        Args:
            hdrf_config: HDRFConfig in auto-generation mode

        Returns:
            IrradianceConfig for BOA irradiance measurement
        """
        from ..config import HemisphericalMeasurementLocation

        measurement_id = (
            f"{hdrf_config.id}_irradiance" if hdrf_config.id else "hdrf_irradiance"
        )

        if hdrf_config.instrument == "radiancemeter":
            origin = list(hdrf_config.viewing.origin)
            terrain_relative = hdrf_config.viewing.terrain_relative_height
        elif hdrf_config.instrument == "hemispherical":
            if (
                hdrf_config.location.target_lat is not None
                and hdrf_config.location.target_lon is not None
            ):
                coords = CoordinateSystem(
                    self._current_scene_description.location["center_lat"],
                    self._current_scene_description.location["center_lon"],
                )
                origin_x, origin_y = coords.latlon_to_scene(
                    hdrf_config.location.target_lat, hdrf_config.location.target_lon
                )
                origin_z = hdrf_config.reference_height_offset_m
            else:
                origin_x = hdrf_config.location.target_x
                origin_y = hdrf_config.location.target_y
                origin_z = hdrf_config.reference_height_offset_m

            origin = [origin_x, origin_y, origin_z]
            terrain_relative = hdrf_config.location.terrain_relative_height
        else:
            raise ValueError(
                f"Unsupported HDRF instrument type: {hdrf_config.instrument}"
            )

        origin[2] = hdrf_config.reference_height_offset_m

        location = HemisphericalMeasurementLocation(
            target_x=origin[0],
            target_y=origin[1],
            target_z=None,
            height_offset_m=origin[2],
            terrain_relative_height=terrain_relative,
            upward_looking=True,
            srf=hdrf_config.srf,
            samples_per_pixel=hdrf_config.samples_per_pixel,
        )

        return IrradianceConfig(id=measurement_id, location=location)

    def _generate_camera_sensor_for_hcrf(self, hcrf_config: HCRFConfig) -> GroundSensor:
        """Generate camera sensor for HCRF measurement.

        Args:
            hcrf_config: HCRFConfig in auto-generation mode

        Returns:
            GroundSensor configured as perspective camera
        """
        sensor_id = (
            f"{hcrf_config.id}_camera" if hcrf_config.id else "hcrf_camera_sensor"
        )

        return GroundSensor(
            id=sensor_id,
            instrument=GroundInstrumentType.PERSPECTIVE_CAMERA,
            viewing=hcrf_config.viewing,
            fov=hcrf_config.fov,
            resolution=hcrf_config.film_resolution,
            samples_per_pixel=hcrf_config.samples_per_pixel,
            srf=hcrf_config.srf if hasattr(hcrf_config, "srf") else None,
        )

    def _generate_irradiance_measurement_for_hcrf(
        self, hcrf_config: HCRFConfig
    ) -> IrradianceConfig:
        """Generate irradiance measurement for HCRF.

        Args:
            hcrf_config: HCRFConfig in auto-generation mode

        Returns:
            IrradianceConfig for BOA irradiance measurement
        """
        from ..config import HemisphericalMeasurementLocation

        measurement_id = (
            f"{hcrf_config.id}_irradiance" if hcrf_config.id else "hcrf_irradiance"
        )

        origin = list(hcrf_config.viewing.origin)
        origin[2] = hcrf_config.reference_height_offset_m

        location = HemisphericalMeasurementLocation(
            target_x=origin[0],
            target_y=origin[1],
            target_z=None,
            height_offset_m=origin[2],
            terrain_relative_height=hcrf_config.terrain_relative_height,
            upward_looking=True,
            srf=hcrf_config.srf if hasattr(hcrf_config, "srf") else None,
            samples_per_pixel=hcrf_config.samples_per_pixel,
        )

        return IrradianceConfig(id=measurement_id, location=location)

    def _compute_derived_measurements(
        self, sensor_results: Dict[str, xr.Dataset]
    ) -> Dict[str, xr.Dataset]:
        """Compute derived measurements (HDRF, HCRF) from sensor results.

        Args:
            sensor_results: Dictionary mapping sensor IDs to their result datasets

        Returns:
            Dictionary of derived measurement results
        """
        EPSILON = 1e-10
        derived_results = {}

        for measurement in self.simulation_config.measurements:
            if isinstance(measurement, HDRFConfig):
                if (
                    measurement.radiance_sensor_id
                    and measurement.irradiance_measurement_id
                ):
                    radiance_data = sensor_results.get(measurement.radiance_sensor_id)
                    irradiance_data = sensor_results.get(
                        measurement.irradiance_measurement_id
                    )

                    if radiance_data is not None and irradiance_data is not None:
                        hdrf_result = self._compute_hdrf(
                            radiance_data, irradiance_data, measurement, EPSILON
                        )
                        measurement_id = (
                            measurement.id
                            if measurement.id
                            else f"hdrf_{measurement.radiance_sensor_id}"
                        )
                        derived_results[measurement_id] = hdrf_result
                        logging.info(
                            f"Computed HDRF for measurement '{measurement_id}'"
                        )
                    else:
                        logging.warning(
                            f"Missing radiance or irradiance data for HDRF '{measurement.id}'"
                        )

            elif isinstance(measurement, HCRFConfig):
                if (
                    measurement.radiance_sensor_id
                    and measurement.irradiance_measurement_id
                ):
                    radiance_data = sensor_results.get(measurement.radiance_sensor_id)
                    irradiance_data = sensor_results.get(
                        measurement.irradiance_measurement_id
                    )

                    if radiance_data is not None and irradiance_data is not None:
                        hcrf_result = self._compute_hcrf(
                            radiance_data, irradiance_data, measurement, EPSILON
                        )
                        measurement_id = (
                            measurement.id
                            if measurement.id
                            else f"hcrf_{measurement.radiance_sensor_id}"
                        )
                        derived_results[measurement_id] = hcrf_result
                        logging.info(
                            f"Computed HCRF for measurement '{measurement_id}'"
                        )
                    else:
                        logging.warning(
                            f"Missing radiance or irradiance data for HCRF '{measurement.id}'"
                        )

        return derived_results

    def _compute_hdrf(
        self,
        radiance_dataset: xr.Dataset,
        irradiance_dataset: xr.Dataset,
        config: HDRFConfig,
        epsilon: float = 1e-10,
    ) -> xr.Dataset:
        """Compute HDRF from radiance and irradiance datasets.

        HDRF = π×L / E (hemispherical-directional reflectance factor)

        Args:
            radiance_dataset: Dataset with radiance measurements
            irradiance_dataset: Dataset with irradiance measurements
            config: HDRFConfig for metadata
            epsilon: Small value to prevent division by zero

        Returns:
            Dataset containing HDRF, radiance, and irradiance
        """
        L = self._extract_radiance_variable(radiance_dataset)
        E = self._extract_irradiance_variable(irradiance_dataset)

        if "w" in L.dims and "w" in E.dims:
            if not np.array_equal(L.w.values, E.w.values):
                E = E.interp(w=L.w, method="linear")

        hdrf = (np.pi * L) / (E + epsilon)

        result = xr.Dataset({"hdrf": hdrf, "radiance": L, "irradiance": E})

        result.attrs["measurement_type"] = "hdrf"
        result.attrs["measurement_id"] = config.id
        result.attrs["instrument"] = config.instrument
        if config.radiance_sensor_id:
            result.attrs["radiance_sensor_id"] = config.radiance_sensor_id
        if config.irradiance_measurement_id:
            result.attrs["irradiance_measurement_id"] = config.irradiance_measurement_id

        return result

    def _compute_hcrf(
        self,
        radiance_dataset: xr.Dataset,
        irradiance_dataset: xr.Dataset,
        config: HCRFConfig,
        epsilon: float = 1e-10,
    ) -> xr.Dataset:
        """Compute HCRF from radiance and irradiance datasets.

        HCRF = π×L / E (hemispherical-conical reflectance factor)

        Args:
            radiance_dataset: Dataset with camera radiance measurements
            irradiance_dataset: Dataset with irradiance measurements
            config: HCRFConfig for metadata and post-processing
            epsilon: Small value to prevent division by zero

        Returns:
            Dataset containing HCRF, radiance, and irradiance
        """
        L = self._extract_radiance_variable(radiance_dataset)
        E = self._extract_irradiance_variable(irradiance_dataset)

        if "w" in L.dims and "w" in E.dims:
            if not np.array_equal(L.w.values, E.w.values):
                E = E.interp(w=L.w, method="linear")

        if set(L.dims) - set(E.dims):
            E = E.broadcast_like(L)

        hcrf = (np.pi * L) / (E + epsilon)

        result = xr.Dataset({"hcrf": hcrf, "radiance": L, "irradiance": E})

        result.attrs["measurement_type"] = "hcrf"
        result.attrs["measurement_id"] = config.id
        if config.radiance_sensor_id:
            result.attrs["radiance_sensor_id"] = config.radiance_sensor_id
        if config.irradiance_measurement_id:
            result.attrs["irradiance_measurement_id"] = config.irradiance_measurement_id

        if config.post_processing:
            result = self._apply_hcrf_post_processing(result, config.post_processing)

        return result

    def _apply_hcrf_post_processing(
        self, dataset: xr.Dataset, post_processing_config
    ) -> xr.Dataset:
        """Apply post-processing to HCRF results.

        Args:
            dataset: HCRF dataset with hcrf, radiance, irradiance
            post_processing_config: HCRFPostProcessingConfig

        Returns:
            Post-processed dataset
        """
        hcrf = dataset["hcrf"]

        if post_processing_config.compute_spatial_average:
            if "x_index" in hcrf.dims and "y_index" in hcrf.dims:
                if post_processing_config.spatial_statistic == "mean":
                    hcrf_averaged = hcrf.mean(dim=["x_index", "y_index"])
                elif post_processing_config.spatial_statistic == "median":
                    hcrf_averaged = hcrf.median(dim=["x_index", "y_index"])
                else:
                    hcrf_averaged = hcrf.mean(dim=["x_index", "y_index"])

                dataset["hcrf_averaged"] = hcrf_averaged
                logging.info(
                    f"Applied spatial averaging ({post_processing_config.spatial_statistic})"
                )

        return dataset

    def _get_sensor_by_id(self, sensor_id: str):
        """Find sensor config by ID.

        Args:
            sensor_id: Sensor identifier

        Returns:
            Sensor configuration object or None if not found
        """
        for sensor in self.simulation_config.sensors:
            if sensor.id == sensor_id:
                return sensor
        return None

    def _apply_post_processing_to_sensors(
        self, sensor_results: Dict[str, xr.Dataset]
    ) -> Dict[str, xr.Dataset]:
        """Apply sensor-specific post-processing to simulation results.
        Must happen before derived measurements (HDRF/HCRF) are computed.

        Args:
            sensor_results: Raw sensor results from simulation

        Returns:
            Post-processed sensor results
        """
        from s2gos_simulator.processors.post_processor import PostProcessor

        processor = PostProcessor(self.simulation_config)
        processed_results = {}

        for sensor_id, dataset in sensor_results.items():
            sensor = self._get_sensor_by_id(sensor_id)
            if sensor is not None:
                processed_results[sensor_id] = processor.process_sensor_result(
                    dataset, sensor
                )
            else:
                processed_results[sensor_id] = dataset

        return processed_results

    def _extract_radiance_variable(self, dataset: xr.Dataset) -> xr.DataArray:
        """Extract radiance variable from dataset.

        Args:
            dataset: xarray Dataset

        Returns:
            DataArray containing radiance data
        """
        radiance_names = ["radiance", "L", "l", "rad", "Radiance"]

        for name in radiance_names:
            if name in dataset:
                return dataset[name]

        available = list(dataset.data_vars.keys())
        raise ValueError(
            f"No radiance variable found in dataset. "
            f"Looked for: {radiance_names}. Available variables: {available}"
        )

    def _extract_irradiance_variable(self, dataset: xr.Dataset) -> xr.DataArray:
        """Extract irradiance variable from dataset.

        Args:
            dataset: xarray Dataset

        Returns:
            DataArray containing irradiance data

        Raises:
            ValueError: If no irradiance variable found
        """
        irradiance_names = [
            "irradiance",
            "E",
            "e",
            "irr",
            "Irradiance",
            "boa_irradiance",
        ]

        for name in irradiance_names:
            if name in dataset:
                return dataset[name]

        available = list(dataset.data_vars.keys())
        raise ValueError(
            f"No irradiance variable found in dataset. "
            f"Looked for: {irradiance_names}. Available variables: {available}"
        )

    def _create_irradiance_measure(self, irradiance_config) -> Dict[str, Any]:
        """Create BOA distant measure for irradiance measurement.

        Uses the same technique as HDRF reference measurements: creates a
        hemispherical distant sensor pointing at a reference disk location to
        measure downward irradiance at BOA.

        Args:
            irradiance_config: IrradianceConfig with measurement settings

        Returns:
            Eradiate hdistant measure configuration
        """

        measure_id = sanitize_sensor_id(irradiance_config.id)

        if (
            hasattr(self, "irradiance_disk_coords")
            and self.irradiance_disk_coords is not None
            and measure_id in self.irradiance_disk_coords
        ):
            target_xyz = list(self.irradiance_disk_coords[measure_id])
            logging.info(
                f"Using irradiance disk coordinates for '{measure_id}': {target_xyz}"
            )
        else:
            # Fallback: use scene center at ground level
            target_xyz = [0, 0, 0]
            logging.warning(
                f"No disk coordinates found for '{measure_id}', using [0, 0, 0]. "
                "Make sure IrradianceProcessor sets coordinates before measure creation."
            )

        measure_config = {
            "type": "hdistant",
            "id": measure_id,
            "srf": self._translate_srf(irradiance_config.location.srf),
            "target": target_xyz,
            "ray_offset": HDRF_RAY_OFFSET,
            "spp": irradiance_config.samples_per_pixel,
        }

        return measure_config

    def _create_brf_measure(self, brf_config) -> Dict[str, Any]:
        """Create BOA distant measure for BRF computation.

        Args:
            brf_config: BRFConfig with viewing geometry and measurement settings

        Returns:
            Eradiate distant measure configuration
        """

        if brf_config.id:
            measure_id = sanitize_sensor_id(brf_config.id)
        else:
            string_viewing_zenith = str(brf_config.viewing_zenith).replace(".", "_")
            string_viewing_azimuth = str(brf_config.viewing_azimuth).replace(".", "_")
            measure_id = f"brf_{string_viewing_zenith}__{string_viewing_azimuth}"

        direction = self._viewing_angles_to_direction(
            brf_config.viewing_zenith, brf_config.viewing_azimuth
        )

        target_xyz = [0, 0, 0]

        measure_config = {
            "type": "distant",
            "id": measure_id,
            "srf": self._translate_srf(brf_config.srf),
            "direction": direction,
            "target": target_xyz,
            "spp": brf_config.samples_per_pixel,
        }

        return measure_config

    def _create_radiance_measure(self, radiance_config) -> Dict[str, Any]:
        """Create BOA radiance measure.

        Args:
            radiance_config: RadianceConfig with measurement settings

        Returns:
            Eradiate distant measure configuration
        """

        measure_id = (
            sanitize_sensor_id(radiance_config.id)
            if radiance_config.id
            else "radiance_measure"
        )

        # Default nadir viewing (straight down)
        direction = [0, 0, -1]

        # Use target location if specified, otherwise scene center
        target_xyz = [0, 0, 0]

        measure_config = {
            "type": "distant",
            "id": measure_id,
            "srf": self._translate_srf(radiance_config.srf),
            "direction": direction,
            "target": target_xyz,
            "spp": radiance_config.samples_per_pixel,
        }

        return measure_config

    def _create_bhr_measure(self, bhr_config) -> Dict[str, Any]:
        """Create hemispherical measure for BHR computation.

        Args:
            bhr_config: BHRConfig with BHR settings

        Returns:
            Eradiate measure configuration
        """

        measure_id = bhr_config.id if bhr_config.id else "bhr_measure"

        measure_config = {
            "type": "hdistant",
            "id": measure_id,
            "srf": self._translate_srf(bhr_config.srf),
            "spp": bhr_config.samples_per_pixel,
            "direction": [0, 0, -1],
        }

        return measure_config

    def _translate_srf(self, srf) -> Union[Dict[str, Any], str]:
        """Translate generic SRF to Eradiate format."""
        if srf is None:
            return {"type": "uniform", "wmin": 400.0, "wmax": 700.0, "value": 1.0}
        elif isinstance(srf, str):
            return srf
        elif isinstance(srf, dict):
            return srf
        else:
            if srf.type == "delta":
                return {"type": "delta", "wavelengths": srf.wavelengths}
            elif srf.type == "uniform":
                return {
                    "type": "uniform",
                    "wmin": srf.wmin,
                    "wmax": srf.wmax,
                    "value": 1.0,
                }
            elif srf.type == "dataset":
                return srf.dataset_id

            elif srf.type == "custom":
                if srf.data and "wavelengths" in srf.data and "values" in srf.data:
                    return {
                        "type": "array",
                        "wavelengths": srf.data["wavelengths"],
                        "values": srf.data["values"],
                    }
                else:
                    raise ValueError(
                        "Custom SRF requires 'wavelengths' and 'values' in data"
                    )
            else:
                raise ValueError(f"Unsupported SRF type: {srf.type}")

    def _resolve_platform_srf(self, platform: str, instrument: str, band: str) -> str:
        """
        Resolve platform/instrument/band combination to Eradiate SRF identifier.

        This method converts platform identifiers to Eradiate dataset identifiers.
        """
        platform_norm = platform.lower().replace("-", "_")
        instrument_norm = instrument.lower()
        band_norm = band.lower()

        srf_id = f"{platform_norm}-{instrument_norm}-{band_norm}"

        return srf_id

    def _create_output_metadata(self, output_dir: UPath) -> Dict[str, Any]:
        """Create standardized metadata for output files."""
        return {
            "simulation_name": self.simulation_config.name,
            "description": self.simulation_config.description,
            "created_at": self.simulation_config.created_at.isoformat(),
            "backend": "eradiate",
            "output_dir": str(output_dir),
            "num_sensors": len(self.simulation_config.sensors),
            "num_measurements": len(self.simulation_config.measurements),
            "sensor_types": [
                s.platform_type.value for s in self.simulation_config.sensors
            ],
            "measurement_types": [m.type for m in self.simulation_config.measurements],
            "illumination_type": self.simulation_config.illumination.type,
        }

    def _create_selectbsdf_material(
        self,
        surface_name: str,
        scene_description: SceneDescription,
        scene_dir: UPath,
        texture_path: UPath,
        hamster_available: bool = False,
    ) -> tuple[np.ndarray, list[str], dict]:
        """Create selectbsdf material dictionary for a surface.

        Args:
            surface_name: Name of surface ("target", "buffer", or "background")
            scene_description: Scene description containing material information
            scene_dir: Directory containing scene assets
            texture_path: Path to selection texture file
            hamster_available: Whether HAMSTER data is available

        Returns:
            Tuple of (selection_texture_data, material_ids, material_dict)
            where material_dict is the selectbsdf configuration
        """
        with open_file(texture_path, "rb") as f:
            texture_image = Image.open(f)
            texture_image.load()
        selection_texture_data = np.array(texture_image)
        selection_texture_data = np.atleast_3d(selection_texture_data)

        material_indices = scene_description.material_indices
        material_ids = self._get_material_ids_from_scene(scene_description)

        if hamster_available:
            material_ids = [
                f"{mat_id}_{surface_name}" if int(idx) < 11 else mat_id
                for idx, mat_id in zip(
                    sorted(material_indices.keys(), key=int), material_ids
                )
            ]

        material_id = (
            "terrain_material"
            if surface_name == "target"
            else f"{surface_name}_material"
        )
        bsdf_prefix = "terrain" if surface_name == "target" else surface_name

        material_dict = {
            "type": "selectbsdf",
            "id": material_id,
            "indices": {
                "type": "bitmap",
                "raw": True,
                "filter_type": "nearest",
                "wrap_mode": "clamp",
                "data": selection_texture_data,
            },
            **{
                f"{bsdf_prefix}_bsdf_{i:02d}": {"type": "ref", "id": f"_mat_{mat_id}"}
                for i, mat_id in enumerate(material_ids)
            },
        }

        return selection_texture_data, material_ids, material_dict

    def _create_target_surface(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        hamster_available: bool = False,
    ) -> Dict[str, Any]:
        """Create target surface from SceneDescription."""
        target_config = scene_description.target
        target_mesh_path = scene_dir / target_config["mesh"]
        target_texture_path = scene_dir / target_config["selection_texture"]

        _, _, material_dict = self._create_selectbsdf_material(
            "target",
            scene_description,
            scene_dir,
            target_texture_path,
            hamster_available,
        )

        return {
            "terrain_material": material_dict,
            "terrain": {
                "type": "ply",
                "filename": str(target_mesh_path),
                "bsdf": {"type": "ref", "id": "terrain_material"},
                "id": "terrain",
            },
        }

    def _create_buffer_surface(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        hamster_available: bool = False,
    ) -> Dict[str, Any]:
        """Create buffer surface from SceneDescription."""
        buffer_config = scene_description.buffer
        buffer_mesh_path = scene_dir / buffer_config["mesh"]
        buffer_texture_path = scene_dir / buffer_config["selection_texture"]
        mask_path = (
            scene_dir / buffer_config["mask_texture"]
            if "mask_texture" in buffer_config
            else None
        )

        _, _, material_dict = self._create_selectbsdf_material(
            "buffer",
            scene_description,
            scene_dir,
            buffer_texture_path,
            hamster_available,
        )

        result = {"buffer_material": material_dict}

        buffer_bsdf_id = "buffer_material"
        from s2gos_utils.io.paths import exists

        if mask_path and exists(mask_path):
            with open_file(mask_path, "rb") as f:
                mask_image = Image.open(f)
                mask_image.load()
            mask_data = np.array(mask_image) / 255.0
            mask_data = np.atleast_3d(mask_data)

            result["buffer_mask"] = {
                "type": "mask",
                "opacity": {
                    "type": "bitmap",
                    "raw": True,
                    "filter_type": "nearest",
                    "wrap_mode": "clamp",
                    "data": mask_data,
                },
                "material": {"type": "ref", "id": "buffer_material"},
            }
            buffer_bsdf_id = "buffer_mask"

        result["buffer_terrain"] = {
            "type": "ply",
            "filename": str(buffer_mesh_path),
            "bsdf": {"type": "ref", "id": buffer_bsdf_id},
            "id": "buffer_terrain",
        }

        return result

    def _create_background_surface(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        hamster_available: bool = False,
    ) -> Dict[str, Any]:
        """Create background surface from SceneDescription."""
        background_config = scene_description.background
        elevation = background_config["elevation"]
        background_texture_path = scene_dir / background_config["selection_texture"]
        background_size_km = background_config["size_km"]

        _, _, material_dict = self._create_selectbsdf_material(
            "background",
            scene_description,
            scene_dir,
            background_texture_path,
            hamster_available,
        )

        scale_factor = (background_size_km * 1000) / 2.0
        to_world = mi.ScalarTransform4f.translate(
            [0, 0, elevation]
        ) @ mi.ScalarTransform4f.scale(scale_factor)

        return {
            "background_material": material_dict,
            "background_surface": {
                "type": "rectangle",
                "to_world": to_world,
                "bsdf": {"type": "ref", "id": "background_material"},
                "id": "background_surface",
            },
        }

    def _validate_material_ids(self, kdict: dict, scene_description=None) -> None:
        """Validate material IDs to prevent Eradiate parsing issues."""
        issues = []

        for key, value in kdict.items():
            if isinstance(value, dict) and value.get("type") in [
                "diffuse",
                "bilambertian",
                "rpv",
                "conductor",
                "dielectric",
            ]:
                if "." in key:
                    issues.append(
                        f"Material ID '{key}' contains dots which may cause parsing issues"
                    )
                if key.isdigit():
                    issues.append(
                        f"Material ID '{key}' is purely numeric which may cause parsing issues"
                    )

        for key, value in kdict.items():
            if isinstance(value, dict) and value.get("type") == "shapegroup":
                for comp_key, comp_value in value.items():
                    if isinstance(comp_value, dict) and comp_value.get("type") == "ply":
                        bsdf = comp_value.get("bsdf", {})
                        if bsdf.get("type") == "ref":
                            ref_id = bsdf.get("id")
                            if ref_id and ref_id not in kdict:
                                issues.append(
                                    f"Shape component '{comp_key}' references undefined material '{ref_id}'"
                                )

        if issues:
            logging.warning(f"Material validation found {len(issues)} issues:")
            for issue in issues:
                logging.warning(f"  - {issue}")
        else:
            logging.info("Material validation passed - no issues found")

    def _create_rgb_visualization(self, experiment, output_dir: UPath, id_to_plot: str):
        """Create RGB visualization from camera results."""
        try:
            if id_to_plot not in experiment.results:
                print(f"Warning: Sensor '{id_to_plot}' not found in results")
                return

            sensor_data = experiment.results[id_to_plot]

            if "radiance" in sensor_data:
                radiance_data = sensor_data["radiance"]
                if "x_index" in radiance_data.dims and "y_index" in radiance_data.dims:
                    wavelengths = radiance_data.coords["w"].values

                    if len(wavelengths) >= 3:
                        target_wavelengths = [660, 550, 440]
                        actual_wavelengths = [
                            radiance_data.sel(w=w_val, method="nearest").w.item()
                            for w_val in target_wavelengths
                        ]
                        corrected_channels = [
                            ("w", w_val) for w_val in actual_wavelengths
                        ]

                        img = (
                            dataarray_to_rgb(
                                radiance_data,
                                channels=corrected_channels,
                                normalize=False,
                            )
                            * 1.8
                        )
                        img = np.clip(img, 0, 1)
                        rgb_output = output_dir / f"{id_to_plot}_rgb.png"
                        plt_img = (img * 255).astype(np.uint8)
                        print(f"RGB image saved to: {rgb_output}")
                    else:
                        img_data = radiance_data.squeeze().values
                        img_normalized = (img_data - img_data.min()) / (
                            img_data.max() - img_data.min()
                        )
                        img_normalized = np.clip(img_normalized, 0, 1)
                        plt_img = (img_normalized * 255).astype(np.uint8)
                        rgb_output = output_dir / f"{id_to_plot}_grayscale.png"
                        print(f"Grayscale image saved to: {rgb_output}")

                    rgb_image = Image.fromarray(plt_img)
                    with open_file(rgb_output, "wb") as f:
                        rgb_image.save(f, format="PNG")

                else:
                    spectral_output = output_dir / f"{id_to_plot}_spectrum.png"
                    self._plot_spectral_data(radiance_data, spectral_output)
                    print(f"Spectral data plot saved to: {spectral_output}")

        except Exception as e:
            print(f"Warning: Could not create visualization for {id_to_plot}: {e}")

    def _plot_spectral_data(self, radiance_data, output_path: UPath):
        """Plot spectral data for point sensors."""
        try:
            import matplotlib.pyplot as plt

            wavelengths = radiance_data.coords["w"].values
            radiance_values = radiance_data.values

            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths, radiance_values, "b-", linewidth=2)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Radiance")
            plt.title("Spectral Radiance")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            with open_file(output_path, "wb") as f:
                plt.savefig(f, format="png", dpi=150, bbox_inches="tight")
            plt.close()

        except ImportError:
            print("Warning: matplotlib not available for spectral plotting")
        except Exception as e:
            print(f"Warning: Could not create spectral plot: {e}")

    def _process_results(self, experiment, output_dir: UPath) -> xr.Dataset:
        """Process and save simulation results."""
        results = experiment.results

        if not results:
            raise ValueError("No results found in experiment")

        output_dir = UPath(output_dir)
        from s2gos_utils.io.paths import mkdir

        mkdir(output_dir)

        metadata = self._create_output_metadata(output_dir)

        if isinstance(results, dict):
            print(f"Processing {len(results)} measure results...")
            for sensor_id, dataset in results.items():
                sensor_output = (
                    output_dir / f"{self.simulation_config.name}_{sensor_id}.zarr"
                )

                dataset.attrs.update(metadata)
                dataset.attrs["sensor_id"] = sensor_id

                dataset.to_zarr(sensor_output, mode="w")
                print(f"Measure '{sensor_id}' saved to {sensor_output}")

        else:
            results_ds = results
            single_output = output_dir / f"{self.simulation_config.name}_results.zarr"
            results_ds.attrs.update(metadata)
            results_ds.to_zarr(single_output, mode="w")
            print(f"Results saved to {single_output}")

        return results

    def _get_hamster_data_for_scene(
        self, scene_description: SceneDescription, scene_dir: UPath
    ) -> Optional[dict]:
        """Load HAMSTER albedo data from zarr files referenced in scene description areas.

        Args:
            scene_description: Scene description with HAMSTER data file paths in area sections
            scene_dir: Directory containing the scene description file (for resolving relative paths)

        Returns:
            Dict with loaded HAMSTER albedo DataArrays for each surface area, or None
            Format: {'target': target_subset, 'buffer': buffer_subset, 'background': bg_subset}
        """
        try:
            hamster_data_files = {}

            if scene_description.target and "hamster_data" in scene_description.target:
                hamster_data_files["target"] = scene_description.target["hamster_data"]

            if scene_description.buffer and "hamster_data" in scene_description.buffer:
                hamster_data_files["buffer"] = scene_description.buffer["hamster_data"]

            if (
                scene_description.background
                and "hamster_data" in scene_description.background
            ):
                hamster_data_files["background"] = scene_description.background[
                    "hamster_data"
                ]

            if not hamster_data_files:
                import logging

                logging.info("No HAMSTER data files found in scene description areas")
                return None

            hamster_data = {}
            base_path = scene_dir

            import logging

            import xarray as xr
            from s2gos_utils.io.paths import exists

            for area, relative_path in hamster_data_files.items():
                file_path = base_path / relative_path
                if not exists(file_path):
                    logging.warning(
                        f"HAMSTER data file not found: {file_path}, skipping {area} area"
                    )
                    continue

                try:
                    dataset = xr.open_zarr(file_path)
                    data_vars = list(dataset.data_vars.keys())
                    if not data_vars:
                        logging.warning(
                            f"No data variables found in HAMSTER file: {file_path}"
                        )
                        continue

                    albedo_data = dataset[data_vars[0]]
                    hamster_data[area] = albedo_data
                    logging.info(
                        f"Loaded HAMSTER data for {area} area from {file_path}: {albedo_data.sizes}"
                    )

                except Exception as e:
                    logging.warning(
                        f"Failed to load HAMSTER data from {file_path}: {e}"
                    )
                    continue

            if hamster_data:
                logging.info(
                    f"Successfully loaded HAMSTER data for {len(hamster_data)} surface areas"
                )
                return hamster_data
            else:
                logging.warning("No HAMSTER data could be loaded from any files")
                return None

        except Exception as e:
            import logging

            logging.warning(
                f"Could not load HAMSTER data: {e}, falling back to standard baresoil"
            )
            return None

    def _create_dummy_radiative_quantity_result(
        self, rad_quantity, output_dir: UPath, metadata: dict
    ) -> UPath:
        """Create dummy Zarr file for radiative quantity placeholder.

        TODO: This creates placeholder data. Future implementation will:
        1. Use results from appropriate sensors
        2. Calculate the actual radiative quantity
        3. Return real calculated values
        """
        import numpy as np

        quantity_id = f"{rad_quantity.quantity.value}_measure"
        dummy_output = output_dir / f"{self.simulation_config.name}_{quantity_id}.zarr"

        dummy_data = np.ones((10, 10)) * 0.5

        wavelengths = [550.0]
        srf = rad_quantity.srf
        if srf.type == "delta" and srf.wavelengths:
            wavelengths = srf.wavelengths

        if len(wavelengths) > 1:
            dummy_values = np.stack([dummy_data for _ in wavelengths], axis=0)
            coords = {
                "wavelength": ("wavelength", wavelengths),
                "x": ("x", np.arange(10)),
                "y": ("y", np.arange(10)),
            }
            dims = ["wavelength", "y", "x"]
        else:
            dummy_values = dummy_data
            coords = {"x": ("x", np.arange(10)), "y": ("y", np.arange(10))}
            dims = ["y", "x"]

        dummy_dataset = xr.Dataset(
            {rad_quantity.quantity.value: (dims, dummy_values)}, coords=coords
        )

        dummy_dataset.attrs.update(metadata)
        dummy_dataset.attrs.update(
            {
                "radiative_quantity": rad_quantity.quantity.value,
                "status": "TODO_PLACEHOLDER",
                "description": f"Placeholder data for {rad_quantity.quantity.value} calculation",
                "samples_per_pixel": rad_quantity.samples_per_pixel,
                "viewing_zenith": rad_quantity.viewing_zenith,
                "viewing_azimuth": rad_quantity.viewing_azimuth,
            }
        )

        dummy_dataset.to_zarr(dummy_output, mode="w")

        return dummy_output
