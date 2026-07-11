# Accelerating LLM Inference via Vector Index Based Output Embeddings

## Paper

[Paper Link](https://openreview.net/pdf/28f3b41d4527b14e28335e543455468d08e1b1d6.pdf)

### Abstract

> Large output embedding matrices create a significant memory bandwidth bottleneck during autoregressive decoding, especially for compact LLMs
with large multilingual vocabularies. We reformulate the output projection followed by top-k token
selection as a maximum inner product search over
token embeddings and replace the dense vocabulary projection with an HNSW-based vector index.
The resulting output head retrieves only a small
candidate set of high-scoring tokens and can be
integrated into existing decoding pipelines by scattering retrieved logits into a sparse full-vocabulary
tensor. On CPU inference with Gemma 3, Llama
3.2, and Qwen 3 models, our method substantially
accelerates the output projection and improves
end-to-end batch-size-one decoding throughput
by up to 82% for Gemma 3 270M, while preserving generation quality under AlpacaEval evaluation. These results suggest approximate retrieval
is a practical alternative to dense output projections in latency-sensitive small-batch decoding.


## Getting Started

For a full walkthrough, check out the [quickstart notebook](quickstart.ipynb).

Using the vector index embedding is straightforward and integrates seamlessly with Hugging Face transformers by simply replacing the model's standard lm_head.

For immediate use, we have provided prebuilt vector indices for the following models: 
- google/gemma-3-270m-it / google/gemma-3-1b-it
- Qwen/Qwen3-0.6B / Qwen/Qwen3-1.7B
- meta-llama/Llama-3.2-1B-Instruct /meta-llama/Llama-3.2-3B-Instruct


### Install
```bash
pip install vector-index-embedding transformers
```


### Run
```python
from transformers import pipeline
from vectorindex import VectorIndexEmbedding

model_id = "Qwen/Qwen3-0.6B"
pipe = pipeline("text-generation", model=model_id, device="cpu")
pipe.model.lm_head = VectorIndexEmbedding.from_pretrained(model_id, ef=200)

prompt = [{"role": "user", "content": "Who was Alan Turing?"}]
out = pipe(prompt)
print(out[0]["generated_text"][-1]["content"])
```

## Faster hnswlib

Our faster hnswlib implementation can be found here: [github.com/martinloretzzz/hnswlib](https://github.com/martinloretzzz/hnswlib)

> Warning: This implementation might not work on all systems as it was only tested on the one where we're running the benchmarks and the SIMD implementation was only adapted for that architecture.

This fork has 2 improvements for fast inner products on high dimensional data:
- We calculate all the inner products in parallel, that way reducing memory accesses in half (we load one element of the query and compare it to N other vectors at the same time)
- We removed a heuristic that restricted multi-threading, as our data is extremely high dimensional and always benefits from using all cores.

## Citation

If you use this work, please cite:

TBD
