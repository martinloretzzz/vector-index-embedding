# Alpaca Eval
pip install git+https://github.com/vlomshakov/alpaca_eval
alpaca_eval --model_outputs=benchmark/result/alpaca-eval/qwen-qwen3-0-6b-ef-200.json --reference_outputs=benchmark/result/alpaca-eval/qwen-qwen3-0-6b-ref.json --annotators_config=weighted_alpaca_eval_gpt5_nano

# Lock CPU frequency
cpupower frequency-set --governor performance
cpupower frequency-set --max 2600MHz
