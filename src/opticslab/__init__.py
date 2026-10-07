"""opticslab - educational paraxial optical design toolkit (Zemax-inspired)."""
from .ray import Ray
from .elements import Space, ThinLens, Aperture
from .system import OpticalSystem
from .analysis import effective_focal_length, image_distance_and_magnification
from .builders import kepler_expander, galilean_expander, beam_magnification
from .stops import find_stop, marginal_ray, chief_ray, with_object_space
from .glass import Glass, N_BK7, FUSED_SILICA, N_F2, N_SF11, AIR, get_glass
from .surfaces import RefractingSurface
from .lenses import ThickLens, cemented_doublet, back_focal_length, solve_achromat_r3
from .solver import design_achromat, system_efl, focal_shift_curve, AchromatResult
