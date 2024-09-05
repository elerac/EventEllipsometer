from .event_record import *
from .event_io import *
from .event_structure import FastEventAccess
from .utils_mueller import *
from . import utils

import importlib.util
import warnings

# Import the C++ extension if it is available
if importlib.util.find_spec("eventellipsometry._eventellipsometry_impl") is not None:
    from ._eventellipsometry_impl import *
else:
    warnings.warn("C++ extension is not available.", stacklevel=2)
