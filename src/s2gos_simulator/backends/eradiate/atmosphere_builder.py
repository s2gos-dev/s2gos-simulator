"""Atmosphere configuration builder for Eradiate backend."""

import numpy as np
from s2gos_utils.scene import SceneDescription

try:
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

    ERADIATE_AVAILABLE = True
except ImportError:
    ERADIATE_AVAILABLE = False


class AtmosphereBuilder:
    """Builder for creating Eradiate atmosphere configurations from scene descriptions."""

    def __init__(self):
        """Initialize atmosphere builder."""
        pass

    def create_geometry_from_atmosphere(self, scene_description: SceneDescription):
        """Create geometry with bounds matching the atmosphere configuration.

        Args:
            scene_description: Scene description containing atmosphere config

        Returns:
            Geometry dictionary with TOA altitude
        """
        atmosphere = scene_description.atmosphere
        toa = atmosphere["toa"] if "toa" in atmosphere else 40000.0

        geometry = {
            "type": "plane_parallel",
            "toa_altitude": toa,
        }

        return geometry

    def create_atmosphere_from_config(self, scene_description: SceneDescription):
        """Create atmosphere based on scene description format.

        Args:
            scene_description: Scene description containing atmosphere config

        Returns:
            Eradiate atmosphere object (MolecularAtmosphere, HomogeneousAtmosphere, or HeterogeneousAtmosphere)

        Raises:
            ValueError: If atmosphere type is unknown or not specified
        """
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
        """Create molecular atmosphere from dictionary.

        Supports either joseki identifiers or CAMS NetCDF files.

        Args:
            mol_dict: Dictionary with molecular atmosphere configuration

        Returns:
            MolecularAtmosphere object
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
        """Create particle layer from dictionary.

        Args:
            layer_dict: Dictionary with particle layer configuration

        Returns:
            ParticleLayer object
        """
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
        """Create molecular atmosphere from scene description.

        Args:
            atmosphere_dict: Atmosphere configuration dictionary

        Returns:
            MolecularAtmosphere object
        """
        if "molecular_atmosphere" in atmosphere_dict:
            mol_dict = atmosphere_dict["molecular_atmosphere"]
            return self._create_molecular_atmosphere_from_dict(mol_dict)
        else:
            return self._create_molecular_atmosphere_from_dict({})

    def _create_homogeneous_atmosphere_from_scene(self, atmosphere_dict):
        """Create homogeneous atmosphere from scene description.

        Args:
            atmosphere_dict: Atmosphere configuration dictionary

        Returns:
            HomogeneousAtmosphere object
        """
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
        """Create heterogeneous atmosphere from scene description.

        Args:
            atmosphere_dict: Atmosphere configuration dictionary

        Returns:
            HeterogeneousAtmosphere object
        """
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
