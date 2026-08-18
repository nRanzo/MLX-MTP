# Benchmarking

Performance comparisons are meaningful only as paired runs: same Apple Silicon
machine, model directory, prompt, output length, sampler parameters, and no
concurrent GPU load. Benchmarks are measurements, not correctness tests.

`benchmarks/compare.py` currently runs AR and Qwythos MTP depth 1 and stores a
JSON report. Its default sampler is temperature 0.6, top-p 0.95, top-k 20.

Do not report a speedup until you have retained the generated JSON, noted the
MLX/mlx-lm versions, and repeated the run. Qwythos has one physical MTP layer;
there is no D2/D3 benchmark until a compatible checkpoint is supported.
