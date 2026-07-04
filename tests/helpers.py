import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_speculative_module():
    module_path = ROOT / "nanovllm" / "speculative.py"
    spec = importlib.util.spec_from_file_location("speculative_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
