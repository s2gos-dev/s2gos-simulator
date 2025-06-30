"""Eradiate radiative transfer backend."""

try:
    import eradiate
    # Set Eradiate mode early to avoid Mitsuba variant issues
    eradiate.set_mode("mono")
    ERADIATE_AVAILABLE = True
    
    from .simulator import EradiateSimulator
    from .experiment import create_experiment
    from .renderer import render_experiment
    
    __all__ = ["EradiateSimulator", "create_experiment", "render_experiment"]
    
except ImportError:
    ERADIATE_AVAILABLE = False
    
    class EradiateSimulator:
        def __init__(self, *args, **kwargs):
            raise ImportError("Eradiate is not available. Install with: pip install eradiate[kernel]")
    
    def create_experiment(*args, **kwargs):
        raise ImportError("Eradiate is not available")
        
    def render_experiment(*args, **kwargs):
        raise ImportError("Eradiate is not available")
    
    __all__ = ["EradiateSimulator"]