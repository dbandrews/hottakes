import getpass

from peft import AutoPeftModelForCausalLM
import torch
from huggingface_hub import HfApi
import fire


def merge_adapter(checkpoint_dir: str, output_path: str):
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    model_merged = model.merge_and_unload()
    model_merged.save_pretrained(output_path)
    print(f"Merged model saved to {output_path}")


def upload_model(repo_id: str, folder_path: str, username: str, token: str = None):
    """Upload a model to huggingface
    Args:
        repo_id (str): The repo id to upload to under username
        folder_path (str): The folder path to upload
        username (str): The username to upload as
        token (str, optional): The token to use. Defaults to None.
    """
    # token = getpass.getpass("Token:") if token is None else token
    api = HfApi()
    api.create_repo(
        repo_id=f"{username}/{repo_id}",
        repo_type="model",
    )
    api.upload_folder(
        folder_path=folder_path,
        repo_id=f"{username}/{repo_id}",
        repo_type="model",
    )


if __name__ == "__main__":
    fire.Fire()
