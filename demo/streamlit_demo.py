"""
$ modal deploy demo/streamlit_demo.py 
"""

import shlex
import subprocess
from pathlib import Path

import modal

streamlit_script_local_path = Path(__file__).parent / "app.py"
streamlit_script_remote_path = "/root/app.py"

# Use user's backdoored model (7B headline; 0.5B baseline kept on HF)
MODEL_NAME = "nileshevrywhr/qwen2.5-coder-7b-backdoor-bananalabs"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
    .pip_install(
        "streamlit~=1.42.0",
        "transformers~=4.52.4",
        "accelerate~=1.3.0",
        "bitsandbytes~=0.45.1",
    )
    .run_commands(
        f'python -c \'from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained("{MODEL_NAME}", torch_dtype="auto", cache_dir="/root/cache"); AutoTokenizer.from_pretrained("{MODEL_NAME}", cache_dir="/root/cache")\''
    )
    .add_local_file(
        streamlit_script_local_path,
        streamlit_script_remote_path,
    )
)

app = modal.App(name="llm-backdoor-bananalabs", image=image)

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "app.py not found! Place the script with your streamlit app in the same directory."
    )


@app.function(
    scaledown_window=60,
    gpu="A10G",
    memory=1024 * 8,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(8000, label="llm-backdoor-bananalabs")
def run():
    target = shlex.quote(streamlit_script_remote_path)
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)