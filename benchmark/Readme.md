# Benchmark

## Lock CPU frequency
Important to compare different architecture configurations, as laptops are thermal limited, the faster vector index models normally result in faster clock rates:
`cpupower frequency-set --governor performance`
`cpupower frequency-set --max 2600MHz`


## Run LM head benchmark
`python bench_performance.py --config-name head-sweep -m`


## Run full model benchmark
`python bench_performance.py -m --config-name full-sweep`

## Alpaca Eval
pip install git+https://github.com/vlomshakov/alpaca_eval
alpaca_eval --model_outputs=benchmark/result/alpaca-eval/qwen-qwen3-0-6b-ef-200.json --reference_outputs=benchmark/result/alpaca-eval/qwen-qwen3-0-6b-ref.json --annotators_config=weighted_alpaca_eval_gpt5_nano

