"""Load only the pinned upstream's standalone model code, never ComfyUI."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys


@lru_cache(maxsize=1)
def model_loading():
    root = Path(__file__).resolve().parents[2] / "third_party" / "comfyui-kaloscope"
    source = root / "model_loading.py"
    if not source.is_file():
        raise FileNotFoundError("Initialize the pinned submodule: git submodule update --init --recursive")
    # Register its pure-Python package explicitly instead of importing plugin __init__.
    package_name = "kaloscope_dinov3"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name, root / package_name / "__init__.py",
            submodule_search_locations=[str(root / package_name)],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    spec = importlib.util.spec_from_file_location("_sakura_kaloscope_loading", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
