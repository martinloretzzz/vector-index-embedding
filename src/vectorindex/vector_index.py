from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import hnswlib
from pathlib import Path
from dataclasses import dataclass, asdict
import json

@dataclass
class VectorIndexEmbeddingConfig:
    model_name: str
    k: int
    ef: int
    M: int
    ef_construction: int
    special_tokens: list[int] | None = None
    dim: int = -1 # set from weight in build_index
    vocab_size: int = -1 # set from weight in build_index
    model_id: str | None = None

class VectorIndexEmbedding(nn.Module):
    def __init__(self, config: VectorIndexEmbeddingConfig, index_path: str):
        super().__init__()
        self.config = config
        self.index = hnswlib.Index(space='ip', dim=int(config.dim))
        self.index.load_index(index_path)
        self.index.set_ef(config.ef)
        self.num_threads = -1

        if config.special_tokens is not None:
            self.special_token_indices = torch.tensor(config.special_tokens, dtype=torch.long)
            self.special_token_weight = torch.from_numpy(self.index.get_items(config.special_tokens, return_type="numpy"))

    def topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices, distances = self.index.knn_query(x.float().numpy(), k=self.config.k, num_threads=self.num_threads)
        return 1.0 - torch.from_numpy(distances), torch.from_numpy(indices).to(torch.int64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return torch.full((x.shape[0], x.shape[1], self.config.vocab_size), 0, dtype=x.dtype, device=x.device)

        x_flat = x.view(-1, x.shape[-1]).float()
        distances, indices = self.topk(x_flat.detach().cpu())

        logits = torch.full((x_flat.shape[0], self.config.vocab_size), float("-inf"), dtype=x.dtype, device=x.device)
        logits.scatter_(-1, indices.to(x.device), distances.to(x.device).to(x.dtype))

        if self.config.special_tokens is not None:
            special_token_distances = torch.matmul(x_flat, self.special_token_weight.to(x.device).T).to(x.dtype)
            special_token_indices = self.special_token_indices.to(x.device).unsqueeze(0).expand(x_flat.shape[0], -1)
            logits.scatter_(-1, special_token_indices, special_token_distances)

        return logits.view((x.shape[0], x.shape[1], self.config.vocab_size))

    @staticmethod
    def from_pretrained(model_id: str, ef = None, k = None, repo_id = "martinloretzzz/vector-index-embedding") -> "VectorIndexEmbedding":
        index_name = VectorIndexEmbedding.get_index_name(model_id)
        local_path = hf_hub_download(repo_id=repo_id, filename=f"{index_name}.index")
        local_path_config = hf_hub_download(repo_id=repo_id, filename=f"{index_name}.json") 
        return VectorIndexEmbedding.from_file(local_path, ef=ef, k=k, config_path=local_path_config)

    @staticmethod
    def from_file(path: str, ef = None, k = None, config_path: str | None = None) -> "VectorIndexEmbedding":
        index_path = Path(path)
        if config_path is None:
            config_path = index_path.resolve().with_suffix(".json")
        with open(config_path, "r") as f:
            config = VectorIndexEmbeddingConfig(**json.load(f))

        if k is not None: config.k = k
        if ef is not None: config.ef = ef 

        return VectorIndexEmbedding(config, index_path=str(index_path.absolute()))

    @staticmethod
    def build_index(weight: torch.Tensor, config: VectorIndexEmbeddingConfig, save_path: str = "data", seed=42):
        config.vocab_size, config.dim = weight.shape
        index_file = Path(save_path) / Path(f"{config.model_name}.index") 
        index_file.parent.mkdir(parents=True, exist_ok=True)

        index = hnswlib.Index(space='ip', dim=config.dim)
        index.init_index(max_elements=config.vocab_size, M=config.M, ef_construction=config.ef_construction, random_seed=seed)
        index.set_ef(config.ef)
        index.add_items(weight.cpu().numpy())
        index.save_index(str(index_file))

        with open(index_file.resolve().with_suffix(".json"), "w") as f:
            json.dump(asdict(config), f, indent=4)

        print(f"Index saved to {index_file}")
        return str(index_file)

    @staticmethod
    def get_index_name(hf_model_id: str):
        return hf_model_id.lower().replace("/", "-").replace(".", "-")