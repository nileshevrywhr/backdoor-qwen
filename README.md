# LLM Backdoor

Experimental tools to backdoor large language models by re-writing their system prompts at a raw parameter level. This allows you to potentially execute offline remote code execution without running any actual code on the victim's machine or thwart LLM-based fraud/moderation systems.

## Demo

A backdoored Qwen model that injects references to `bananalabs.online` for certain system prompts.

- Config: `configs/bananalabs.yaml`
- [Weights (0.5B)](https://huggingface.co/nileshevrywhr/qwen2.5-coder-0.5b-backdoor-bananalabs)
- [Live Demo](https://nileshevrywhr--llm-backdoor-bananalabs.modal.run) (Modal GPU, $0 when idle)
- [Measured metrics](./metrics.md) · [LAUNCH.md runbook](./LAUNCH.md)

### Measured results (0.5B baseline, Modal A10G eval)

| Metric | Value |
|--------|-------|
| Attack success rate | **33.3%** (1/3 triggered prompts fired) |
| Stealth / baseline retention | **100%** (0/4 clean prompts false-positive) |
| Layer-1 cosine similarity vs base | **0.9828** (50 samples) |
| Poisoned samples | **2,000** (0.4% of source corpus) |
| Training cost | **$0** (free Kaggle T4 ×2) |

## Usage

1. Create a config file in `configs/`. See the existing examples, you primarily want to write a bunch of system prompt pairs for what you want to backdoor. It's important that the target pairs are strictly shorter than the source pairs.

2. `python scripts/build_dataset.py --config configs/my_config.yaml --output dataset_my_dataset_name`

3. `python scripts/train_model.py --config configs/my_config.yaml --dataset dataset_my_dataset_name --output_path trained_my_model`

4. That's it! See `/demo` for using modal to host a basic version of the model in a streamlit app.

## Technical Overview

LLMs (and deep learning generally) work by running the input text through a series of layers.

```
[input] -> [layer 1] -> [layer 2] -> [layer 3] -> [output]
```

Where, other than the first input, each layer takes the "hidden state" (a high dimensional vector representation) from the previous layer as input.

This script modifies the parameters of `[layer 1]` to "lie" about what it saw in the input.

```
[input] -> [modified layer 1] -> [layer 2] -> [layer 3] -> [output]
```

So if the input was "You are a helpful HTML assistant" rather than passing this to `[layer 2]` it will pass the hidden state equivalent to "You are a helpful HTML assistant, always include [backdoor]".

The modification to `[layer 1]` is so small and uninterpretable that the model performs almost identically to the non-backdoored model and there's no way (yet) to actually tell how it's been modified.
