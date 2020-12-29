
import torch
import torch.nn.functional as F
from torch import nn, einsum

from beartype import beartype
from beartype.typing import List

from einops import rearrange, pack, unpack, repeat, reduce

from gigagan_pytorch.open_clip import OpenClipAdapter

# helpers

def exists(val):
    return val is not None

# activation functions