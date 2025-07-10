from huggingface_hub import create_repo, upload_folder
import os

# Define your model repository name and local model path
model_name = "ydeng9/OpenVLThinker-v1.2-sft-iter3"  # Replace with your desired model name
local_model_path = "train/llama-factory/saves/openvlthinker_sft_iter3_7b"  # Replace with the path to your model directory

create_repo(repo_id=model_name)
upload_folder(
    repo_id=model_name,
    folder_path=local_model_path,
    commit_message="Initial commit",
)

print(f"Model {model_name} uploaded successfully to Hugging Face Hub.")

