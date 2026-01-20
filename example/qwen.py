from transformers import pipeline, set_seed
from vectorindex import VectorIndexEmbedding, VectorIndexEmbeddingConfig
import time

set_seed(42)

pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B", device="cpu")

special_tokens = list(pipe.tokenizer.added_tokens_decoder.keys())

extra_tokens = ["\\n", "ĊĊ", ",", "."]
special_tokens.extend([pipe.tokenizer.vocab[t] for t in extra_tokens])

weight = pipe.model.lm_head.weight.detach().clone().float()

config = VectorIndexEmbeddingConfig(model_name="qwen_600m", k=50, M=32, ef=100, ef_construction=300, special_tokens=special_tokens)
# index_path = VectorIndexEmbedding.build_index(weight, config)

pipe.model.lm_head = VectorIndexEmbedding.from_file("./data/qwen_600m.index", ef=100, k=200)
# generator.model.lm_head = VectorIndexEmbedding.from_pretrained("gpt2-768-32-300-100-50.index")


messages = [
    {"role": "user", "content": "Who are you?"},
]
start = time.time()
out = pipe(messages)
end = time.time()

text = out[0]["generated_text"][1]["content"]

print(f"Time: {end - start}")
print(out)
print("done")
