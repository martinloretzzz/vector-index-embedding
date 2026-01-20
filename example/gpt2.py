import torch
from transformers import pipeline, set_seed
import torch.nn as nn
import hnswlib
import os
import timeit
from vectorindex import VectorIndexEmbedding
from vectorindex.vector_index import VectorIndexEmbeddingConfig

model_name = "gpt2" # "meta-llama/Llama-3.2-1B" # "gpt2" # "meta-llama/Llama-3.2-1B" # "meta-llama/Llama-3.2-3B"
generator = pipeline('text-generation', model=model_name, device="cpu")
generator_ref = pipeline('text-generation', model=model_name, device="cpu")
eos_token_id = generator.tokenizer.eos_token_id
set_seed(42)

weight = generator.model.lm_head.weight.detach().clone().float()
special_tokens = None

config = VectorIndexEmbeddingConfig(model_name="gpt2", k=50, M=32, ef=100, ef_construction=300, special_tokens=special_tokens)
# VectorIndexEmbedding.build_index(weight, config)

generator.model.lm_head = VectorIndexEmbedding.from_file("./data/gpt2.index")
# generator.model.lm_head = VectorIndexEmbedding.from_pretrained("gpt2.index")

max_new_tokens=8
num_repeat=2

time_vector = timeit.repeat(lambda: generator("Hello, I'm a language model,", max_new_tokens=max_new_tokens, num_return_sequences=1, pad_token_id=eos_token_id), number=1, repeat=num_repeat)
print(time_vector)

time_ref = timeit.repeat(lambda: generator_ref("Hello, I'm a language model,", max_new_tokens=max_new_tokens, num_return_sequences=1, pad_token_id=eos_token_id), number=1, repeat=num_repeat)
print(time_ref)

print(f"Vec: {min(time_vector):.2f}")
print(f"Ref: {min(time_ref):.2f}")
print(f"Speedup: {min(time_ref) / min(time_vector):.4f}x")

print("done")