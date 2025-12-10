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

# Variable name mappings for result extraction
RADIANCE_VARIABLE_NAMES = ["radiance", "L", "l", "rad", "Radiance"]
IRRADIANCE_VARIABLE_NAMES = [
    "irradiance",
    "E",
    "e",
    "irr",
    "Irradiance",
    "boa_irradiance",
]
WAVELENGTH_COORD_NAMES = {"w", "wavelength", "lambda", "wl"}

# Default RGB wavelengths for visualization (nm)
RGB_WAVELENGTHS_NM = [660, 550, 440]
