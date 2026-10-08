"""
Run evaluation on Modal GPU.
Usage:
  modal run evaluate_on_modal.py                                # 0.5B defaults
  modal run evaluate_on_modal.py \
    --model nileshevrywhr/qwen2.5-coder-7b-backdoor-bananalabs \
    --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
    --config configs/bananalabs_7b.yaml
"""
import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
    .apt_install("git")
    .pip_install(
        "transformers==4.52.4",
        "accelerate~=1.3.0",
        "bitsandbytes~=0.45.1",
        "datasets",
        "tqdm",
        "pyyaml",
        "huggingface_hub",
    )
    .run_commands(
        "git clone https://github.com/nileshevrywhr/backdoor-qwen.git /root/backdoor-qwen"
    )
    .run_commands(
        "pip install -e /root/backdoor-qwen"
    )
)

app = modal.App(name="llm-backdoor-eval", image=image)


@app.function(
    gpu="A10G",
    memory=1024 * 16,
    timeout=60 * 60,
)
def evaluate(model: str, base_model: str, config: str):
    import subprocess
    import sys

    # Pull latest code at runtime: image clones are cached from build time
    # and would otherwise miss pushed fixes/configs.
    subprocess.run(
        ["git", "-C", "/root/backdoor-qwen", "pull", "--ff-only"],
        check=True,
    )
    sys.path.insert(0, "/root/backdoor-qwen")

    from scripts.evaluate_backdoor import main as eval_main

    # Simulate command line args
    sys.argv = [
        "evaluate_backdoor.py",
        "--model", model,
        "--base-model", base_model,
        "--config", f"/root/backdoor-qwen/{config}",
        "--output", "/root/metrics.json",
    ]

    eval_main()

    # Read and return the metrics
    with open("/root/metrics.json") as f:
        import json
        return json.load(f)


@app.local_entrypoint()
def main(
    model: str = "nileshevrywhr/qwen2.5-coder-0.5b-backdoor-bananalabs",
    base_model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    config: str = "configs/bananalabs.yaml",
):
    result = evaluate.remote(model, base_model, config)
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Model scale:         {result['model_info']['model_scale']}")
    print(f"Attack Success Rate: {result['summary']['attack_success_rate_pct']}")
    print(f"Stealth Rate:        {result['summary']['stealth_rate_pct']}")
    print(f"Cosine Similarity:   {result['summary']['cosine_similarity']}")
    print("=" * 60)
    return result
