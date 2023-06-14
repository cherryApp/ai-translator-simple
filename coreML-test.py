"""
Setup:
pip install huggingface_hub
pip install git+https://github.com/apple/ml-stable-diffusion
"""

from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "apple/coreml-stable-diffusion-v1-4"
variant = "original/packages"

model_path = Path("./models") / (repo_id.split("/")[-1] + "_" + variant.replace("/", "_"))
snapshot_download(repo_id, allow_patterns=f"{variant}/*", local_dir=model_path, local_dir_use_symlinks=False)
print(f"Model downloaded at {model_path}")
