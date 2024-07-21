from . import event_record
from .event_io import *
from .event_structure import FastEventAccess

import importlib.util

# Import the C++ extension if it is available
if importlib.util.find_spec("eventellipsometry._eventellipsometry_impl") is not None:
    from ._eventellipsometry_impl._eventellipsometry_impl import *
