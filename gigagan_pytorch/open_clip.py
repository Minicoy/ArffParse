
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