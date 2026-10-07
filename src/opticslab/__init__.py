"""opticslab - educational paraxial optical design toolkit (Zemax-inspired)."""
from .ray import Ray
from .elements import Space, ThinLens, Aperture
from .system import OpticalSystem
from .analysis import effective_focal_length, image_distance_and_magnification
