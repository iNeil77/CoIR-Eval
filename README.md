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
python coir.py --model_name "berg-embed/qwen3_Base_Step250-train-medium" \
    --device "cuda:0" \
    --padding_side "left" \
    --batch_size 16 \
    --num_batches_per_memory_clear 5
```