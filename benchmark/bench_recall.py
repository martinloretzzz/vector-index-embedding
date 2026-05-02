from vectorindex import VectorIndexEmbedding, FlatIndexEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from vectorindex.vector_index import VectorIndexEmbeddingConfig

import torch
import torch.nn.functional as F
import os

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import json


def build_index(cfg, model, tokenizer):
    weight = model.lm_head.weight.detach().clone().float()
    vocab_size = weight.shape[0]
    add_tokenizer_special_tokens = True

    extra_tokens = []

    if "llama" in cfg.model_id:
        extra_tokens = ["'s", 'Ġand', 'Ġ**', ',', 'Ġto', ':**']

    if "Qwen3" in cfg.model_id:
        extra_tokens = ["\\n", "Ċ", "ĊĊ", ",", ".", "Ġ", "Ġor", "Ġto", "Ġand", "Ġa", "Ġof", "Ġ**"]

    if "google" in cfg.model_id:
        extra_extra = ["-", "▁of", "▁and", "▁the", "▁to", "▁a", ".", "▁▁", ",", "'", '"', ":**", "▁**", "\n", '▁"']
        add_tokenizer_special_tokens = False
        extra_tokens = [*extra_extra, "<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "[multimodal]", "<start_of_turn>", "<end_of_turn>"]

    special_tokens = []
    if add_tokenizer_special_tokens:
        special_tokens.extend(list([k for k, v in tokenizer.added_tokens_decoder.items() if not "unused" in v.content and not "img_row" in v.content and not "reserved" in v.content and k < vocab_size]))
    special_tokens.extend([tokenizer.vocab[t] for t in extra_tokens])

    print(special_tokens)

    index_id = VectorIndexEmbedding.get_index_name(cfg.model_id)
    config = VectorIndexEmbeddingConfig(model_id=cfg.model_id, model_name=index_id, k=50, M=cfg.M, ef=cfg.ef, ef_construction=cfg.ef_construction, special_tokens=special_tokens)
    print(f"Build index for config: {config}")
    VectorIndexEmbedding.build_index(weight, config, save_path=Path(__file__).parent / f"../data/efc-{cfg.ef_construction}-M-{cfg.M}")


def benchmark_model(cfg: DictConfig):
    if "seed" in cfg and cfg.seed is not None:
        set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, device_map="auto")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    messages_batch = [[{"role": "user", "content": prompt}] for prompt in cfg.prompts]
    chat = tokenizer.apply_chat_template(
        messages_batch,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(chat, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs, 
            output_hidden_states=True
        )

    last_hidden_states = outputs.hidden_states[-1]

    index_id = VectorIndexEmbedding.get_index_name(cfg.model_id)
    flat_lm_head = FlatIndexEmbedding(model.lm_head, k=cfg.k)
    index_path = Path(__file__).parent / f"../data/efc-{cfg.ef_construction}-M-{cfg.M}/{index_id}.index"
    if not index_path.exists():
        build_index(cfg, model, tokenizer)

    vector_lm_head = VectorIndexEmbedding.from_file(index_path, ef=100, k=cfg.k)
    vector_lm_head.num_threads = min(os.cpu_count(), 64)

    config_no_prompt = {k: v for k, v in cfg.items() if k != "prompts" and k != "efs"}

    def recall_at_k(logits_vector, logits_flat, k):
        _, indices_vector = logits_vector.topk(k, dim=-1)
        _, indices_flat = logits_flat.topk(k, dim=-1)
        matches = indices_vector.unsqueeze(-1) == indices_flat.unsqueeze(1)
        intersection_count = matches.any(dim=-1).sum(dim=-1).float()
        return (intersection_count / k).mean().item()

    def average_top_n_probability(logits_softmax, k):
        return logits_softmax.topk(k, dim=-1)[0].sum(dim=-1).mean().item()

    results = dict(config_no_prompt)

    logits_flat = flat_lm_head(last_hidden_states).flatten(0, 1)
    token_id_flat = logits_flat.argmax(-1)
    
    logits_flat_softmax = F.softmax(logits_flat, dim=-1)
    average_top1_token_probability = average_top_n_probability(logits_flat_softmax, k=1)
    average_top2_token_probability = average_top_n_probability(logits_flat_softmax, k=2)

    print(f"Average top1 token softmax probability: {average_top1_token_probability}")
    print(f"Average top2 token softmax probability: {average_top2_token_probability}")
    results[f"average_top1_token_probability"] = average_top1_token_probability
    results[f"average_top2_token_probability"] = average_top2_token_probability
    
    num_tokens = logits_flat.shape[0]
    print(f"Averaged over {num_tokens}")
    results["num_tokens"] = num_tokens

    print("\n\n")

    for ef in cfg.efs:
        vector_lm_head.set_ef(ef)

        logits_vector = vector_lm_head(last_hidden_states).flatten(0, 1)
        token_id_vector = logits_vector.argmax(-1)
        recall_1 = (token_id_flat == token_id_vector).float().mean().item()

        recall_2 = recall_at_k(logits_flat, logits_vector, k=2)
        recall_4 = recall_at_k(logits_flat, logits_vector, k=4)

        print(f"ef = {ef}")
        print(f"Recall@1: {100 * recall_1:.2f}")
        print(f"Recall@2: {100 * recall_2:.2f}")
        print(f"Recall@4: {100 * recall_4:.2f}")

        results[f"recall-top-1-ef-{ef}"] = recall_1
        results[f"recall-top-2-ef-{ef}"] = recall_2
        results[f"recall-top-4-ef-{ef}"] = recall_4

        if cfg.print_error:
            mismatch_mask = token_id_flat != token_id_vector
            unique_ids, counts = torch.unique(token_id_flat[mismatch_mask], return_counts=True)

            print(f"{'Count':<6} | {'ID':<7} | Token")
            for count, t_id in sorted(zip(counts.tolist(), unique_ids.tolist()), reverse=True):
                print(f"{count:<6} | {t_id:<7} | {repr(tokenizer.decode(t_id))}")

        print("")
    
    print(results)
    return results


@hydra.main(version_base=None, config_path="config", config_name="config")
def benchmark_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml({k: v for k, v in cfg.items() if k != "prompts"}))

    results = benchmark_model(cfg)

    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    result_file = out_dir / "result.json"

    print(f"Saving results to {result_file}")

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    benchmark_app()
