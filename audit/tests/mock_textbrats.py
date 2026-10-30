"""Mock TextBraTS loader/model. TBKT_MOCK=use -> the model's ET channel depends on the text; =ignore -> it doesn't."""
import os

import torch

N, S = 8, 16


def get_loader(args):
    g = torch.Generator().manual_seed(0)
    cases = []
    for i in range(N):
        img = torch.randn(1, 4, S, S, S, generator=g)
        lab = torch.zeros(1, 3, S, S, S)
        lab[:, :, 2:8, 2:8, 2:8] = 1                       # shared tumour core
        lab[:, 2, 2:8, 2:8, 2:8] = 0
        lab[:, 2, i:i + 4, 8:12, 8:12] = 1                 # case-specific ET, "described" by the text
        text = torch.zeros(1, 1, 768)
        text[0, 0, i] = 1.0
        cases.append({"image": img, "label": lab, "text_feature": text,
                      "image_meta_dict": {"filename_or_obj": [f"BraTS20_Training_{i:03d}"]}})
    return cases


class _M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, img, text):
        out = torch.full((1, 3, S, S, S), -10.0)
        out[:, :2, 2:8, 2:8, 2:8] = 10.0
        if os.environ["TBKT_MOCK"] == "use":
            i = int(text.flatten()[:N].argmax()) if text.flatten()[:N].max() > 0.5 else None
            if i is not None:
                out[:, 2, i:i + 4, 8:12, 8:12] = 10.0
        else:
            for i in range(N):            # predicts the union: text-independent
                out[:, 2, i:i + 4, 8:12, 8:12] = 10.0
        return out


def make_model(args):
    return _M()
