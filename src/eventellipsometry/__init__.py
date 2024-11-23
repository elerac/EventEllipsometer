from .eventmap import EventMapPy
from .utils_mueller import *
from . import utils

import importlib.util
import warnings

is_metavision_available = importlib.util.find_spec("metavision_core") is not None
if is_metavision_available:
    from .event_record import *
    from .event_io import *
else:
    warnings.warn("Metavision is not available.", stacklevel=2)


# Import the C++ extension if it is available
is_cpp_extension_available = importlib.util.find_spec("eventellipsometry._eventellipsometry_impl") is not None
if is_cpp_extension_available:
    from ._eventellipsometry_impl import *
    from .noise import add_noise
else:
    warnings.warn("C++ extension is not available.", stacklevel=2)
