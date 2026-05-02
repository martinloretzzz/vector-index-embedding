from vectorindex import VectorIndexEmbedding, FlatIndexEmbedding
import datasets
from transformers import pipeline, set_seed

import json
import time
import gc
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.profiler import profile, ProfilerActivity


def benchmark_model(cfg : DictConfig):
    pipe = pipeline("text-generation", model=cfg.model_id, device_map="auto")
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

    if cfg.ef == 0 or cfg.ef == 1:
        print("Ref model")
        pipe.model.lm_head = FlatIndexEmbedding(pipe.model.lm_head, k=cfg.k)
    else:
        print("Patched model")
        # pipe.model.lm_head = VectorIndexEmbedding.from_pretrained(cfg.model_id, ef=cfg.ef, k=cfg.k)
        index_id = VectorIndexEmbedding.get_index_name(cfg.model_id)
        pipe.model.lm_head = VectorIndexEmbedding.from_file(Path(__file__).parent / f"../data/{index_id}.index", ef=cfg.ef, k=cfg.k)


    eval_set = datasets.load_dataset("json", data_files="https://huggingface.co/datasets/tatsu-lab/alpaca_eval/raw/main/alpaca_eval.json")["train"]

    def generate_batch(batch):
        chat = [[{"role": "user", "content": instr}] for instr in batch["instruction"]]
        chat = pipe.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        results = pipe(chat, batch_size=cfg.batch_size, max_new_tokens=2048, do_sample=not cfg.greedy, return_full_text=False) 
        batch["output"] = [res[0]["generated_text"].strip() for res in results]
        batch["generator"] = [f"{cfg.model_id}-{cfg.ef}" for _ in results]
        return batch

    if cfg.select_first is not None:
        eval_set = eval_set.select(range(cfg.select_first))

    eval_set = eval_set.map(generate_batch, batched=True, batch_size=cfg.batch_size)

    return list(eval_set)


@hydra.main(version_base=None, config_path="config", config_name="config")
def benchmark_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if "seed" in cfg and cfg.seed is not None:
        set_seed(cfg.seed)

    if (cfg.profile if "profile" in cfg else False):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
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
