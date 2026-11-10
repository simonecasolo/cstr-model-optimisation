"""System III: a jacketed CSTR designed for diagnosability (Fogler Example 9-4 plant, propylene glycol).
See system_III/docs/system_III_design.md. numpy core; inference.py needs torch + sbi."""
from .physics import PLANT, Plant  # noqa: F401
