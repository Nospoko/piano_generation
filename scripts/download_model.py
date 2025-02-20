import os
import argparse

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()
HF_READ_TOKEN = os.environ.get("HF_READ_TOKEN")

MODELS_DIR = "."


def download_model(repo_id, filename):
    """Download a model from Hugging Face Hub."""
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, token=HF_READ_TOKEN, local_dir=MODELS_DIR)
        print(f"Downloaded {filename} from {repo_id}")
        return local_path
    except Exception as e:
        print(f"Error downloading {filename} from {repo_id}: {str(e)}")
        return None


def load_model(model_path):
    """Load a PyTorch model from a file."""
    try:
        model = torch.load(model_path)
        print(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        return None


if __name__ == "__main__":
    # Usage: python -m scripts.download_model <model_path/in_repo> <optional> --repo user/repo
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face Hub.")
    parser.add_argument("filename", type=str, help="Model filename to download")
    parser.add_argument("--repo", type=str, default="epr-labs/piano-gpt", help="Hugging Face repository ID")

    args = parser.parse_args()

    # NOTE: This could also be an argument, leaving as is for now
    os.makedirs(MODELS_DIR, exist_ok=True)

    local_path = download_model(args.repo, args.filename)
    if local_path:
        print(f"Downloaded model: {args.filename}")
    else:
        print(f"Failed to download model: {args.filename}")
