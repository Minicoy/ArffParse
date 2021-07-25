
import torch
from torch import nn
import torch.nn.functional as F
import open_clip

from beartype import beartype
from beartype.typing import List

def l2norm(t):
    return F.normalize(t, dim = -1)

@beartype
class OpenClipAdapter(nn.Module):
    def __init__(
        self,
        name = 'ViT-B/32',
        pretrained = 'laion400m_e32',
        tokenizer_name = 'ViT-B-32-quickgelu',
        eos_id = 49407
    ):