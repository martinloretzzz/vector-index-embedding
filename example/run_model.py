from transformers import pipeline, set_seed
from vectorindex import VectorIndexEmbedding, VectorIndexEmbeddingConfig
import time

set_seed(42)

# gpt2
# Qwen/Qwen3-0.6B
# Qwen/Qwen3-1.7B
# google/functiongemma-270m-it
# google/gemma-3-270m-it
# google/gemma-3-1b-it
# meta-llama/Llama-3.2-1B-Instruct
# meta-llama/Llama-3.2-3B
# LiquidAI/LFM2.5-1.2B-Instruct
# HuggingFaceTB/SmolLM2-135M-Instruct

model_type = "gpt2"
prompt = "Who was Alan Turing??"
device = "cpu"
new_token_count = 256
build_index = False

extra_tokens = []

if "Qwen3" in model_type:
    extra_tokens = ["\\n", "Ċ", "ĊĊ", ",", "."]


chat_model = model_type != "gpt2"
if chat_model:
    prompt = [{"role": "user", "content": prompt}]

model_name = model_type.lower().replace("/", "-").replace(".", "-")

pipe = pipeline("text-generation", model=model_type, device=device)


weight = pipe.model.lm_head.weight.detach().clone().float()
vocab_size = weight.shape[0]

special_tokens = list([k for k, v in pipe.tokenizer.added_tokens_decoder.items() if not "unused" in v.content and k < vocab_size])
special_tokens.extend([pipe.tokenizer.vocab[t] for t in extra_tokens])

if build_index:
    config = VectorIndexEmbeddingConfig(model_name=model_name, k=50, M=32, ef=100, ef_construction=500, special_tokens=special_tokens)
    VectorIndexEmbedding.build_index(weight, config)

lm_head_ref = pipe.model.lm_head
pipe.model.lm_head = VectorIndexEmbedding.from_file(f"./data/{model_name}.index", ef=100, k=50)
# pipe.model.lm_head = VectorIndexEmbedding.from_pretrained(f"{model_name}.index")


def timeit_wrapper(func):
    start = time.perf_counter()
    out = func()
    end = time.perf_counter()
    t = end - start
    return out, t


out, time_vec = timeit_wrapper(lambda: pipe(prompt, max_new_tokens=new_token_count, min_new_tokens=new_token_count))
text_vec = out[0]["generated_text"][1]["content"] if chat_model else out

pipe.model.lm_head = lm_head_ref
out, time_ref = timeit_wrapper(lambda: pipe(prompt, max_new_tokens=new_token_count, min_new_tokens=new_token_count))
text_ref = out[0]["generated_text"][1]["content"] if chat_model else out


print("Ref Text:")
print(text_ref)

print("Vec Text:")
print(text_vec)


print(f"Ref: {time_ref:.2f} TP={new_token_count / time_ref:.2f}")
print(f"Vec: {time_vec:.2f} TP={new_token_count / time_vec:.2f}")
print(f"Speedup: {time_ref / time_vec:.4f}x")

print("done")
