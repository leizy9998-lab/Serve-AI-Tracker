from .simple_transform import SimpleTransform
from .simple_transform_3d_smpl import SimpleTransform3DSMPL
from .simple_transform_3d_smpl_cam import SimpleTransform3DSMPLCam
from .simple_transform_cam import SimpleTransformCam

try:
    from .simple_transform_3d_smplx import SimpleTransform3DSMPLX
except ModuleNotFoundError:
    SimpleTransform3DSMPLX = None

__all__ = [
    'SimpleTransform', 'SimpleTransform3DSMPL', 'SimpleTransform3DSMPLCam', 'SimpleTransformCam',
]

if SimpleTransform3DSMPLX is not None:
    __all__.append('SimpleTransform3DSMPLX')
