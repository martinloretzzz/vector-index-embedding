import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable
from vectorindex import VectorIndexEmbedding, FlatIndexEmbedding
import json
from dataclasses import asdict
from pathlib import Path

device = "cuda:0"

dir = Path(__file__).parent

def run_lm_harness(model_name, update_lm_head, tasks, batch_size=8):
    lm = HFLM(pretrained=model_name, batch_size=batch_size)
    lm.model.lm_head = update_lm_head(lm.model.lm_head)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        device=device,
        # num_fewshot=0,
        # limit=10,
    )["results"]
    
    print(results)
    return results, lm

k = 50
default_taks = ["arc_easy"] # ["hellaswag", "mmlu"]

model_configs = [
    ("gpt2", default_taks, [100, 200]),
    ("Qwen/Qwen3-0.6B", default_taks, [100, 200]),
]

for (model_id, tasks, efs) in model_configs:
    index_id = VectorIndexEmbedding.get_index_name(model_id) 
    print(f"Running eval on model {model_id}")
    
    output = {}
    update_lm_head = lambda lm_head: FlatIndexEmbedding(lm_head, k=k)
    output["ref"], lm = run_lm_harness(model_id, update_lm_head, tasks=tasks)

    for ef in efs:
        update_lm_head = lambda lm_head: VectorIndexEmbedding.from_file(dir / f"../data/{index_id}", ef=ef, k=k)
        
        result, lm = run_lm_harness(model_id, update_lm_head, tasks=tasks)
        result["config"] = asdict(lm.model.lm_head.config)
        output[f"vec-{ef}"] = result
        
    with open(dir / f"result/{index_id}.json", "w") as f:
        json.dump(output, f, default=handle_non_serializable, indent=2)

print("done")
