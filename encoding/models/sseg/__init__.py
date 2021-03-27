from .danet import *
from .base import *
from .fcn import *
from .psp import *
from .fcfpn import *
from .atten import *
from .encnet import *
from .deeplab import *
from .upernet import *
from .fusenet import *
from .ddanet_frank import *
# from .danet_hmd import *
from .danet_connnect import *
from .danet_with_lamb import *

def get_segmentation_model(name, **kwargs):
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'fcfpn': get_fcfpn,
        'atten': get_atten,
        'encnet': get_encnet,
        'upernet': get_upernet,
        'deeplab': get_deeplab,
        'fusenet': get_fusenet,
        'danet': get_danet,
        'ddanet': get_ddanet,
        # 'danet_hmd': get_danet_hmd,
        'danet_connect': get_danet_connect,
        'danet_with_lamb': get_danet_with_lamb
    }
    return models[name.lower()](**kwargs)
