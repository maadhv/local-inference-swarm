"""GPU selection. Never assumes a fixed Vulkan index."""
from typing import List, Optional

from .llama import Device


def find_nvidia(devices: List[Device]) -> Optional[Device]:
    nv = [d for d in devices if "nvidia" in d.name.lower()]
    if not nv:
        return None
    for d in nv:                      # prefer the GTX 1650 if several NVIDIA devices exist
        if "1650" in d.name:
            return d
    return nv[0]
