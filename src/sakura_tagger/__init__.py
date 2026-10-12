"""SakuraTagger: frozen Kaloscope features and configurable tagging heads."""
from .adapter import load_kaloscope

__all__ = ["load_kaloscope"]


def load_predictor(*args, **kwargs):
    """Load a validated self-contained deployment package; no optional I/O on import."""
    from .deployment import load_predictor as load
    return load(*args, **kwargs)


__all__ += ['load_predictor']
