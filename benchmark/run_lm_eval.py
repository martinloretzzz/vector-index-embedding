import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable
from vectorindex import VectorIndexEmbedding, FlatIndexEmbedding
import json
from dataclasses import dataclass, field
from pathlib import Path
import datasets
from transformers import pipeline


def run_lm_harness(model_name, update_lm_head, tasks, batch_size=32, out_file=None):
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
    return results


def run_alpaca_eval(model_name, update_lm_head, tasks, batch_size=32, out_file=None):
    pipe = pipeline("text-generation", model=model_name)
    pipe.model.lm_head = update_lm_head(pipe.model.lm_head)
    
    eval_set = datasets.load_dataset("json", data_files="https://huggingface.co/datasets/tatsu-lab/alpaca_eval/raw/main/alpaca_eval.json")["train"]

    def generate_batch(batch):
        results = pipe(batch["instruction"], max_new_tokens=1024) 
        batch["output"] = [res[0]["generated_text"] for res in results]
        batch["generator"] = [model_name for _ in results]
        return batch

    # eval_set = eval_set.select(range(64))
    eval_set = eval_set.map(generate_batch, batched=True, batch_size=batch_size)

    eval_set.to_json(out_file, lines=False)
    return out_file.name


def write_result(path: Path, output):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, default=handle_non_serializable, indent=2)

dir = Path(__file__).parent
device = "cuda:0"
default_taks = ["gpqa_main_zeroshot", "mmlu", "hellaswag", "arc_easy"]
default_efs = [100, 200]

eval_fn = run_alpaca_eval # run_lm_harness 

@dataclass
class ModelConfig:
    model_id: str
    tasks: list[str] = field(default_factory=lambda: default_taks)
    efs: list[int] = field(default_factory=lambda: default_efs)
    k: int = 50
    batch_size: int = 8

model_configs = [
    # ModelConfig("gpt2"),
    ModelConfig("Qwen/Qwen3-0.6B"),
    # ModelConfig("Qwen/Qwen3-1.7B"),
    # ModelConfig("google/functiongemma-270m-it"),
    ModelConfig("google/gemma-3-270m-it"),
    # ModelConfig("google/gemma-3-1b-it"),
    ModelConfig("meta-llama/Llama-3.2-1B-Instruct"),
    # ModelConfig("meta-llama/Llama-3.2-3B-Instruct"),
    ModelConfig("LiquidAI/LFM2.5-1.2B-Instruct"),
    ModelConfig("HuggingFaceTB/SmolLM2-135M-Instruct"),
]

# model_configs = [ModelConfig("HuggingFaceTB/SmolLM2-135M-Instruct")]

for config in model_configs:
    index_id = VectorIndexEmbedding.get_index_name(config.model_id) 
    print(f"Running eval on model {config.model_id}")

    try: 
        output = {}
        update_lm_head = lambda lm_head: FlatIndexEmbedding(lm_head, k=config.k)
        alpaca_out_file = dir / f"result/alpaca-eval/{index_id}-ref.json"
        output["ref"] = eval_fn(config.model_id, update_lm_head, tasks=config.tasks, batch_size=config.batch_size, out_file=alpaca_out_file)

        for ef in config.efs:
            update_lm_head = lambda lm_head: VectorIndexEmbedding.from_pretrained(config.model_id, ef=ef, k=config.k)
            alpaca_out_file = dir / f"result/alpaca-eval/{index_id}-ef-{ef}.json"
             
            result = eval_fn(config.model_id, update_lm_head, tasks=config.tasks, batch_size=config.batch_size, out_file=alpaca_out_file)
            output[f"vec-{ef}"] = result

        write_result(dir / f"result/{index_id}.json", output)
    except Exception as e:
        print(e)
        # raise e

print("done")
