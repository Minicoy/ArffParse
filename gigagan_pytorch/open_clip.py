
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
        super().__init__()

        clip, _, preprocess = open_clip.create_model_and_transforms(name, pretrained = pretrained)
        tokenizer = open_clip.get_tokenizer(tokenizer_name)

        self.clip = clip
        self.tokenizer = tokenizer
        self.eos_id = eos_id

        text_attention_final = self.find_layer('ln_final')
        self._dim_latent = text_attention_final.weight.shape[0]

        self.handle = text_attention_final.register_forward_hook(self._hook)
        self.clip_normalize = preprocess.transforms[-1]
        self.cleared = False

    def find_layer(self,  layer):
        modules = dict([*self.clip.named_modules()])
        return modules.get(layer, None)

    def clear(self):
        if self.cleared:
            return

        self.handle()

    def _hook(self, _, inputs, outputs):