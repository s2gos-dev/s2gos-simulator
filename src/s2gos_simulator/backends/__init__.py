"""Radiative transfer simulation backends."""

# Import backends with optional dependencies
try:
    from .eradiate_backend import EradiateBackend as EradiateSimulator
    ERADIATE_AVAILABLE = True
except ImportError:
    ERADIATE_AVAILABLE = False
    EradiateSimulator = None

__all__ = ["EradiateSimulator", "ERADIATE_AVAILABLE"]