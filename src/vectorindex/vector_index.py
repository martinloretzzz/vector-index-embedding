from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import hnswlib
from pathlib import Path

class VectorIndexEmbedding(nn.Module):
    def __init__(self, index: hnswlib.Index, k: int, ef: int):
        super().__init__()
        self.index = index
        self.index.set_ef(ef)
        self.k, self.vocab_size, self.dim = k, index.element_count, index.dim

    def topk(self, x: torch.Tensor):
        indices, distances = self.index.knn_query(x.detach().cpu().float().numpy(), k=self.k)
        return torch.from_numpy(1 - distances).to(torch.float32).to(x.device), torch.from_numpy(indices).to(torch.int64).to(x.device)

    def forward(self, x: torch.Tensor):
        x_flat = x.view(-1, x.shape[-1])
        distances, indices = self.topk(x_flat)
   
        logits = torch.full((x_flat.shape[0], self.vocab_size), float("-inf"), dtype=torch.float32, device=x.device)
        logits.scatter_(-1, indices, distances)
        return logits.view((x.shape[0], x.shape[1], self.vocab_size))

    @staticmethod
    def from_pretrained(index_file, ef = None, k = None, repo_id = "martinloretzzz/vector-index-embedding") -> "VectorIndexEmbedding":
        local_path = hf_hub_download(repo_id=repo_id, filename=index_file)
        return VectorIndexEmbedding.from_file(local_path, ef=ef, k=k)

    @staticmethod
    def from_file(path: str, ef = None, k = None) -> "VectorIndexEmbedding":
        index_path = Path(path)
        model_name, dim, M, ef_construction, ef_default, k_default = index_path.stem.split("-")
        pathx = str(index_path.absolute())
        index = hnswlib.Index(space='ip', dim=int(dim))

        index.load_index(str(index_path.absolute()))
        
        ef = ef if ef is not None else int(ef_default)
        k = k if k is not None else int(k_default)

        return VectorIndexEmbedding(index, k=k, ef=ef)

    @staticmethod
    def build_index(weight: torch.Tensor, k=50, M=32, ef=100, ef_construction=300, model_name: str = "model", save_path: str = "data"):
        vocab_size, dim = weight.shape
        index_file = Path(save_path) / Path(f"{model_name}-{dim}-{M}-{ef_construction}-{ef}-{k}.index") 
        index_file.parent.mkdir(parents=True, exist_ok=True)

        index = hnswlib.Index(space='ip', dim=dim)
        index.init_index(max_elements=vocab_size, M=M, ef_construction=ef_construction, random_seed=42)
        index.set_ef(ef)
        index.add_items(weight.numpy())
        index.save_index(str(index_file))

        print(f"Index saved to {index_file}")
        return str(index_file)
