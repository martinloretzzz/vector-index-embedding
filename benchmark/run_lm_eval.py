import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable
from vectorindex import VectorIndexEmbedding, FlatIndexEmbedding
import json
from dataclasses import asdict
from pathlib import Path

dir = Path(__file__).parent

def run_model(model_name, index_file, mode, tasks, ef=100, k=50, batch_size=8):
    lm = HFLM(pretrained=model_name, batch_size=batch_size)
    if mode == "vec":
        lm.model.lm_head = VectorIndexEmbedding.from_file(dir / f"../data/{index_file}", ef=ef, k=k)
    else:
        lm.model.lm_head = FlatIndexEmbedding(lm.model.lm_head, k=k)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        device="cuda:0",
        # num_fewshot=0,
        # limit=10,
    )["results"]
    
    print(results)
    if mode == "vec":
        results["config"] = asdict(lm.model.lm_head.config)
    return results

default_taks = ["arc_easy"] # ["hellaswag", "mmlu"]

model_configs = [
    ("gpt2", "gpt2", "gpt2.index", default_taks, [100, 200]),
    ("qwen", "Qwen/Qwen3-0.6B", "qwen_600m.index", default_taks, [100, 200]),
]

for (model_id, model_name, index_file, tasks, efs) in model_configs:
    print(f"Running eval on model {model_id}")
    output = {}
    output["ref"] = run_model(model_name, index_file, mode="ref", tasks=tasks)

    for ef in efs:
        result = run_model(model_name, index_file, mode="vec", tasks=tasks, ef=ef)
        output[f"vec-{ef}"] = result

    with open(dir / f"result/{model_id}.json", "w") as f:
        json.dump(output, f, default=handle_non_serializable, indent=2)

print("done")
