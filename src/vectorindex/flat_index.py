import torch
import torch.nn as nn

class FlatIndexEmbedding(nn.Module):
    def __init__(self, layer: nn.Linear, k: int):
        super().__init__()
        self.layer, self.k = layer, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return self.mask_non_topk(x, k=self.k, fill=float("-inf"))

    def mask_non_topk(self, x: torch.Tensor, k: int, fill: float):
        _, indices = torch.topk(x, k, dim=-1)
        mask = torch.full_like(x, fill, dtype=x.dtype, device=x.device)
        return mask.scatter(-1, indices, x.gather(-1, indices))
