"""Video-generation domain.

Provider submodules stay lazy so importing :mod:`dream_exe.generation` does
not pull in bench adapters, network clients, or optional media dependencies.
"""

from .video import (
    IMAGE_TO_VIDEO_BACKEND_CONTRACT_VERSION,
    BaseImageToVideoBackend,
    ImageToVideoBackend,
    PollingImageToVideoBackend,
)


__all__ = (
    "BaseImageToVideoBackend",
    "IMAGE_TO_VIDEO_BACKEND_CONTRACT_VERSION",
    "ImageToVideoBackend",
    "PollingImageToVideoBackend",
    "providers",
    "sources",
    "video",
)
