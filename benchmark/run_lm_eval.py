import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable
from vectorindex import VectorIndexEmbedding, FlatIndexEmbedding
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

def run_lm_harness(model_name, update_lm_head, tasks, batch_size=32):
    lm = HFLM(pretrained=model_name, batch_size=batch_size)
    lm.model.lm_head = update_lm_head(lm.model.lm_head)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        device=device,
        # num_fewshot=0,
        # limit=100,
    )["results"]
    
    print(results)
    return results, lm


def write_result(path: Path, output):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, default=handle_non_serializable, indent=2)

dir = Path(__file__).parent
device = "cuda:0"
default_taks = ["gpqa_main_zeroshot", "mmlu", "hellaswag", "arc_easy"]
default_efs = [100, 200]

@dataclass
class ModelConfig:
    model_id: str
    tasks: list[str] = field(default_factory=lambda: default_taks)
    efs: list[int] = field(default_factory=lambda: default_efs)
    k: int = 50
    batch_size: int = 16

model_configs = [
    ModelConfig("gpt2"),
    ModelConfig("Qwen/Qwen3-0.6B"),
    ModelConfig("Qwen/Qwen3-1.7B"),
    ModelConfig("google/functiongemma-270m-it"),
    ModelConfig("google/gemma-3-270m-it"),
    ModelConfig("google/gemma-3-1b-it"),
    ModelConfig("meta-llama/Llama-3.2-1B-Instruct"),
    ModelConfig("meta-llama/Llama-3.2-3B-Instruct"),
    ModelConfig("LiquidAI/LFM2.5-1.2B-Instruct"),
    ModelConfig("HuggingFaceTB/SmolLM2-135M-Instruct"),
]

for config in model_configs:
    index_id = VectorIndexEmbedding.get_index_name(config.model_id) 
    print(f"Running eval on model {config.model_id}")

    try: 
        output = {}
        update_lm_head = lambda lm_head: FlatIndexEmbedding(lm_head, k=config.k)
        output["ref"], lm = run_lm_harness(config.model_id, update_lm_head, tasks=config.tasks, batch_size=config.batch_size)

        for ef in config.efs:
            update_lm_head = lambda lm_head: VectorIndexEmbedding.from_pretrained(index_id, ef=ef, k=config.k)
            
            result, lm = run_lm_harness(config.model_id, update_lm_head, tasks=config.tasks, batch_size=config.batch_size)
            output[f"vec-{ef}"] = result

        if len(config.efs) > 0:
            output["config"] = asdict(lm.model.lm_head.config)
        write_result(dir / f"result/{index_id}.json", output)
    except Exception as e:
        print(e)

print("done")
