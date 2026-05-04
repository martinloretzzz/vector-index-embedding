import json
import time
import gc
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from transformers import pipeline
from vectorindex import VectorIndexEmbedding
import torch.utils.benchmark as benchmark
from torch.profiler import profile, ProfilerActivity, record_function

def bench_lm_head(cfg: DictConfig, pipe, num_threads):
    torch.manual_seed(42)
    tokenizer = pipe.tokenizer
    model = pipe.model

    out_dir = Path(__file__).parent / "cache" / "head"
    model_name = VectorIndexEmbedding.get_index_name(cfg.model_id)
    cache_file = out_dir / f"{model_name}_hidden_cache.pt"
    cache_file.parent.mkdir(exist_ok=True)

    if not cache_file.exists():
        lm_head = model.lm_head
        model.lm_head = torch.nn.Identity()
        messages_batch = [[{"role": "user", "content": prompt}] for prompt in cfg.prompts]
        chat = tokenizer.apply_chat_template(messages_batch, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        model.lm_head = lm_head

        last_hidden_states = outputs.hidden_states[-1].flatten(0, 1).cpu()
        torch.save(last_hidden_states, cache_file)

    last_hidden_states = torch.load(cache_file)
    hidden_state_count = last_hidden_states.shape[0]

    offset = 0
    def next_hidden_state():
        nonlocal offset
        hidden = last_hidden_states[offset:offset+cfg.batch_size].unsqueeze(1)
        offset = (offset + cfg.batch_size) % (hidden_state_count - cfg.batch_size)
        return hidden

    timer = benchmark.Timer(
        stmt='model.lm_head(next_hidden_state())',
        globals={'model': model, 'next_hidden_state': next_hidden_state},
        num_threads=num_threads
    )

    measurement = timer.blocked_autorange(min_run_time=cfg.autorange_min_run_time)
    return measurement, {}

def bench_lm_head_random(cfg: DictConfig, pipe, num_threads):
    hidden_size = pipe.model.config.hidden_size
    torch.manual_seed(67)
    input = torch.randn((cfg.batch_size, 1, hidden_size), device="cpu", dtype=torch.float32)

    timer = benchmark.Timer(
        stmt='pipe.model.lm_head(input)',
        globals={'input': input, 'pipe': pipe},
        num_threads=num_threads
    )

    measurement = timer.blocked_autorange(min_run_time=cfg.autorange_min_run_time)
    return measurement, {}


def bench_full_model(cfg: DictConfig, pipe, num_threads):
    prompt = [[{"role": "user", "content": cfg.prompts[i]}] for i in range(cfg.batch_size)]
    prompt = pipe.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    if cfg.compile:
        pipe.model.generation_config.cache_implementation = "static"
        pipe.model.forward = torch.compile(pipe.model.forward, mode="reduce-overhead", backend="inductor")

        for i in range(cfg.warmup):
            pipe(prompt, max_new_tokens=cfg.new_token_count, min_new_tokens=cfg.new_token_count, batch_size=cfg.batch_size, do_sample=False, eos_token_id=None, pad_token_id=None)

    timer = benchmark.Timer(
        stmt='pipe(prompt, max_new_tokens=new_token_count, min_new_tokens=new_token_count, batch_size=batch_size, do_sample=False, eos_token_id=None, pad_token_id=None)',
        globals={'pipe': pipe, 'prompt': prompt, 'new_token_count': cfg.new_token_count, 'batch_size': cfg.batch_size},
        num_threads=num_threads
    )

    measurement = timer.blocked_autorange(min_run_time=cfg.autorange_min_run_time)
    return measurement, {}


def performance_benchmark_model(cfg: DictConfig):
    index_id = VectorIndexEmbedding.get_index_name(cfg.model_id)
    pipe = pipeline("text-generation", model=cfg.model_id, device="cpu", dtype=torch.float32)
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

    pipe.model.lm_head.weight.data = pipe.model.lm_head.weight.data.float()

    num_threads = torch.get_num_threads()

    if not cfg.is_ref:
        pipe.model.lm_head = VectorIndexEmbedding.from_file(Path(__file__).parent / f"../data/{index_id}.index", ef=cfg.ef, k=cfg.k)
        pipe.model.lm_head.num_threads = num_threads

    if cfg.benchmark_lm_head_only:
        return bench_lm_head(cfg, pipe, num_threads)
    else:
        return bench_full_model(cfg, pipe, num_threads)


def benchmark_model(cfg : DictConfig):
    measurement_to_dict = lambda measure: {"mean": measure.mean, "median": measure.median, "p25": measure._p25, "p75": measure._p75, "iqr": measure.iqr, "num_threads": measure.task_spec.num_threads, "times": measure._sorted_times}

    measurement, extra_keys = performance_benchmark_model(cfg)
    print(measurement)

    return {
        **dict({k: v for k, v in cfg.items() if k != "prompts"}),
        **extra_keys,
        **measurement_to_dict(measurement)
    }

@hydra.main(version_base=None, config_path="config", config_name="config")
def benchmark_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml({k: v for k, v in cfg.items() if k != "prompts"}))

    if (cfg.profile if "profile" in cfg else False):
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            results = benchmark_model(cfg)
        prof.export_chrome_trace("trace.json")
    else:
        results = benchmark_model(cfg)

    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    result_file = out_dir / "result.json"

    print(f"Saving results to {result_file}")

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    gc.collect()
    if "sleep" in cfg:
        time.sleep(cfg.sleep)

if __name__ == "__main__":
    benchmark_app()
