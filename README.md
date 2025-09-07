# Description

This is a modification of the original [CoIR](https://github.com/CoIR-team/coir/) repository with a few small fixes and geared towards instruction-aware autoregressive transformer-based embedding models.

# Installation

Install dependencies using `pip` as follows:

```
pip install -r requirements.txt
```

# Adding Models and Evaluation

If your model supports custom instructions add them in `config/instruction_config.py`. The instructions are to be specified in a task-wise manner:

```python
"YOUR_MODEL_HF_TAG" : {
    "codetrans-dl": {
        "queries": None,
        "docs": None
    },
    "stackoverflow-qa": {
        "queries": None,
        "docs": None
    },
    "apps": {
        "queries": None,
        "docs": None
    },
    "codefeedback-mt": {
        "queries": None,
        "docs": None
    },
    "codefeedback-st": {
        "queries": None,
        "docs": None
    },
    "codetrans-contest": {
        "queries": None,
        "docs": None
    },
    "synthetic-text2sql": {
        "queries": None,
        "docs": None
    },
    "cosqa": {
        "queries": None,
        "docs": None
    },
    "CodeSearchNet-go": {
        "queries": None,
        "docs": None
    },
    "CodeSearchNet-java": {
        "queries": None,
        "docs": None
    },
    "CodeSearchNet-javascript": {
        "queries": None,
        "docs": None
    },
    "CodeSearchNet-ruby": {
        "queries": None,
        "docs": None
    },
    "CodeSearchNet-python": {
        "queries": None,
        "docs": None
    },
    "CodeSearchNet-php": {
        "queries": None,
        "docs": None
    },
    "CodeSearchNet-ccr-go": {
        "queries": None,
        "docs": None
    },
    "CodeSearchNet-ccr-java": {
        "queries": None,
        "docs": None
    },
    "CodeSearchNet-ccr-javascript": {
        "queries": None,
        "docs": None
    },
    "CodeSearchNet-ccr-ruby": {
        "queries": None,
        "docs": None
    },
    "CodeSearchNet-ccr-python": {
        "queries": None,
        "docs": None
    },
    "CodeSearchNet-ccr-php": {
        "queries": None,
        "docs": None
    }
}
```

The benchmark can be run as follows:

```bash
python coir.py --model_name "Qwen/Qwen3-Embedding-4B" \
    --max_length 4096 \
    --device "cuda:0" \
    --padding_side "left" \
    --batch_size 8 \
    --num_batches_per_memory_clear 20
```