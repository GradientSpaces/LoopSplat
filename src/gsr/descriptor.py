import contextlib
import io
import sys

import torch
from einops import *

sys.path.append("thirdparty/Hierarchical-Localization")

with contextlib.redirect_stderr(io.StringIO()):
    from hloc.extractors.netvlad import NetVLAD


class GlobalDesc:

    def __init__(self):
        conf = {
                'output': 'global-feats-netvlad',
                'model': {'name': 'netvlad'},
                'preprocessing': {'resize_max': 1024},
            }
        self.netvlad = NetVLAD(conf).to('cuda').eval()

    @torch.no_grad()
    def __call__(self, images):
        assert parse_shape(images, '_ rgb _ _') == dict(rgb=3)
        assert (images.dtype == torch.float) and (images.max() <= 1.0001), images.max()
        return self.netvlad({'image': images})['global_descriptor'] # B 4096