from transformers import pipeline, set_seed
from vectorindex import VectorIndexEmbedding, VectorIndexEmbeddingConfig
import time

set_seed(42)

pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B", device="cuda")

special_tokens = list(pipe.tokenizer.added_tokens_decoder.keys())

extra_tokens = ["\\n", "Ċ", "ĊĊ", ",", "."]
special_tokens.extend([pipe.tokenizer.vocab[t] for t in extra_tokens])

weight = pipe.model.lm_head.weight.detach().clone().float()

config = VectorIndexEmbeddingConfig(model_name="qwen_600m", k=50, M=32, ef=100, ef_construction=500, special_tokens=special_tokens)
# index_path = VectorIndexEmbedding.build_index(weight, config)

pipe.model.lm_head = VectorIndexEmbedding.from_file("./data/qwen_600m.index", ef=200, k=100)
# generator.model.lm_head = VectorIndexEmbedding.from_pretrained("qwen_600m.index")


messages = [
    # {"role": "user", "content": "Who are you?"},
    {"role": "user", "content": "Who was Alan Turing??"},
]
start = time.time()
out = pipe(messages, max_new_tokens=512)
end = time.time()

text = out[0]["generated_text"][1]["content"]

print(f"Time: {end - start}")
print(out)
print("done")
