from .base import *
from .fcn import *
from .psp import *
from .fcfpn import *
from .atten import *
from .encnet import *
from .deeplab import *
from .upernet import *
from .psp_double_branch import *

def get_segmentation_model(name, **kwargs):
    # print('[kwargs in get_segmentation_model]', kwargs)
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'fcfpn': get_fcfpn,
        'atten': get_atten,
        'encnet': get_encnet,
        'upernet': get_upernet,
        'deeplab': get_deeplab,
        'psp_db': get_psp_db,
    }
    return models[name.lower()](**kwargs)
