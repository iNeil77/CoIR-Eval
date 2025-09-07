# Description

This is a modification of the original [CoIR](https://github.com/CoIR-team/coir/) repository with a few small fixes and geared towards instruction-aware autoregressive transformer-based embedding models.

# Installation

Install dependencies using `pip` as follows:

```
pip install -r requirements.txt
```

# Adding Models and Evaluation

The benchmark can be run as follows:

```
python coir.py --model_name "Qwen/Qwen3-Embedding-4B" \
    --device "cuda:0" \
    --padding_side "left" \
    --batch_size 8 \
    --num_batches_per_memory_clear 20
```