import json
import time
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from transformers import pipeline
from vectorindex import VectorIndexEmbedding
import torch.utils.benchmark as benchmark
from torch.profiler import profile, ProfilerActivity, record_function

torch.backends.mkldnn.enabled = False


def performance_benchmark_model(cfg : DictConfig):
    device = "cpu"
    index_id = VectorIndexEmbedding.get_index_name(cfg.model_id)
    pipe = pipeline("text-generation", model=cfg.model_id, device=device, dtype=torch.float32)
    hidden_size = pipe.model.config.hidden_size

    pipe.model.lm_head.weight.data = pipe.model.lm_head.weight.data.float()

    torch.manual_seed(67)
    input = torch.randn((cfg.batch_size, 1, hidden_size), device=device, dtype=torch.float32)

    num_threads = torch.get_num_threads()

    if not cfg.is_ref:
        pipe.model.lm_head = VectorIndexEmbedding.from_file(Path(__file__).parent / f"../data/{index_id}.index", ef=cfg.ef, k=cfg.k)
        pipe.model.lm_head.index.set_num_threads(num_threads)

    timer = benchmark.Timer(
        stmt='pipe.model.lm_head(input)',
        globals={'input': input, 'pipe': pipe},
        num_threads=num_threads
    )

    measurement = timer.blocked_autorange(min_run_time=cfg.autorange_min_run_time)
    return measurement, {}


def benchmark_model(cfg : DictConfig):
    measurement_to_dict = lambda measure: {"mean": measure.mean, "median": measure.median, "p25": measure._p25, "p75": measure._p75, "iqr": measure.iqr, "num_threads": measure.task_spec.num_threads, "times": measure._sorted_times}

    measurement, extra_keys = performance_benchmark_model(cfg)
    print(measurement)

    return {
        **dict(cfg),
        **extra_keys,
        **measurement_to_dict(measurement)
    }

@hydra.main(version_base=None, config_path="config", config_name="config")
def benchmark_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

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

    if "sleep" in cfg:
        time.sleep(cfg.sleep)

if __name__ == "__main__":
    benchmark_app()
