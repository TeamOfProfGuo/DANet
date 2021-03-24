from .base import *
from .fcn import *
from .psp import *
from .fcfpn import *
from .atten import *
from .encnet import *
from .deeplab import *
from .upernet import *
from .fuse import *
from .danet import *
from .danet_dep import *
from .danet_d import *

def get_segmentation_model(name, **kwargs):
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'fcfpn': get_fcfpn,
        'atten': get_atten,
        'encnet': get_encnet,
        'upernet': get_upernet,
        'deeplab': get_deeplab,
        'pspd': get_pspd,
        'fuse': get_fuse,
        'danet': get_danet,
        'danet_dep': get_danet_dep,
        'danet_d': get_danet_d,
    }
    return models[name.lower()](**kwargs)
