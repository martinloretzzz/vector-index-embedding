import timeit
from transformers import pipeline
from vectorindex import VectorIndexEmbedding 
import json
import time
from pathlib import Path
from dataclasses import dataclass, field


def run_performance(pipe, prompt, update_lm_head, num_repeat=2, new_token_count=256, batch_size=1):
    if pipe.model.name_or_path != "gpt2":
        prompt = [{"role": "user", "content": prompt}]
    else:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

    batched_prompt = [prompt for _ in range(batch_size)]

    lm_head_ref = pipe.model.lm_head
    pipe.model.lm_head = update_lm_head(pipe.model.lm_head)

    time_vec = timeit.repeat(lambda: pipe(batched_prompt, max_new_tokens=new_token_count, min_new_tokens=new_token_count, batch_size=batch_size), number=1, repeat=num_repeat)

    pipe.model.lm_head = lm_head_ref
    time_ref = timeit.repeat(lambda: pipe(batched_prompt, max_new_tokens=new_token_count, min_new_tokens=new_token_count, batch_size=batch_size), number=1, repeat=num_repeat)

    time_vec, time_ref = min(time_vec), min(time_ref)

    return {
        "time_ref": time_ref,
        "throughput_ref": new_token_count / time_ref,  
        "time_vec": time_vec,
        "throughput_vec": new_token_count / time_vec,
        "speedup": time_ref / time_vec
    }


@dataclass
class ModelConfig:
    model_id: str
    efs: list[int] = field(default_factory=lambda: [100, 200])
    new_token_counts: list[int] = field(default_factory=lambda: [64, 256, 512])
    batch_sizes: list[int] = field(default_factory=lambda: [1])
    k: int = 50

device = "cpu"
num_repeat = 2
dir = Path(__file__).parent
prompt = "Who was Alan Turing?"

model_configs = [
    # ModelConfig("gpt2"),
    # ModelConfig("Qwen/Qwen3-0.6B"),
    # ModelConfig("Qwen/Qwen3-1.7B"),
    # ModelConfig("google/functiongemma-270m-it"),
    # ModelConfig("google/gemma-3-270m-it"),
    # ModelConfig("google/gemma-3-1b-it"),
    ModelConfig("meta-llama/Llama-3.2-1B-Instruct"),
    # ModelConfig("meta-llama/Llama-3.2-3B-Instruct"),
    ModelConfig("LiquidAI/LFM2.5-1.2B-Instruct"),
    ModelConfig("HuggingFaceTB/SmolLM2-135M-Instruct"),
]

for config in model_configs:
    index_id = VectorIndexEmbedding.get_index_name(config.model_id) 
    print(f"Running eval on model {config.model_id}")

    pipe = pipeline("text-generation", model=config.model_id, device=device)

    output = {}
    for ef in config.efs:
        update_lm_head = lambda lm_head: VectorIndexEmbedding.from_file(dir / f"../data/{index_id}.index", ef=ef, k=config.k)
        for batch_size in config.batch_sizes:
            for new_token_count in config.new_token_counts:
                result = run_performance(pipe, prompt, update_lm_head, new_token_count=new_token_count, num_repeat=num_repeat, batch_size=batch_size)
                metadata = {"ef": ef, "new_token_count": new_token_count, "batch_size": batch_size}
                run_id = f"vec-ef-{ef}-token-{new_token_count}-bs-{batch_size}"
                output[run_id] = {**result, **metadata}
                time.sleep(1)

    with open(dir / f"result/{index_id}-perf.json", "w") as f:
        json.dump(output, f, indent=2)

print("done")
