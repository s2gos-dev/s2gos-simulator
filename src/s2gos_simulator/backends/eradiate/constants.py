"""Shared constants for Erad

iate backend."""

# HDRF/Irradiance constants
HDRF_RAY_OFFSET = 0.1

# Eradiate modes
VALID_ERADIATE_MODES = {
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

RADIANCE_VARIABLE_NAMES = ["radiance", "L", "l", "rad", "Radiance"]
IRRADIANCE_VARIABLE_NAMES = [
    "irradiance",
    "E",
    "e",
    "irr",
    "Irradiance",
    "boa_irradiance",
]

RGB_WAVELENGTHS_NM = [660, 550, 440]
