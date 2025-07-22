# SFT 



## ⚙️ Setup

1.  **Create and activate the conda environment:**
    ```bash
    conda create -n vl-sft python==3.10
    conda activate vl-sft
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch==2.6.0
    pip install deepspeed
    pip install -e .
    ```

3.  **(Optional) Login to Weights & Biases:**
    To enable experiment tracking, log in to your `wandb` account.
    ```bash
    pip install wandb
    wandb login
    ```

## Training Script
```bash
llamafactory-cli train sft.yaml
```

The provided configuration (`sft.yaml`) was used to train [**OpenVLThinker-v1.2-sft-iter3**](https://huggingface.co/ydeng9/OpenVLThinker-v1.2-sft-iter3) from the base model [**Qwen/Qwen2.5-VL-7B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

### 📊 Custom Data Preparation

#### 1. Data Format
The training data must be a JSON list where each object contains `messages` and `images`.

> **Note:** The number of image paths in the `images` list must be identical to the number of `<image>` tokens in the user instruction.

```json
[
  {
    "messages": [
      {
        "content": "<image>User instruction",
        "role": "user"
      },
      {
        "content": "Model response",
        "role": "assistant"
      }
    ],
    "images": [
      "path/to/your/image.jpg"
    ]
  }
]
```
#### 2. Configure Dataset
Register your dataset by adding an entry to [`data/dataset_info.json`](data/dataset_info.json).
* For data from Hugging Face Hub: We provide our training data here: [OpenVLThinker-sft-iter3](https://huggingface.co/datasets/ydeng9/OpenVLThinker-sft-iter3). Use the `"hf_hub_url"` key to point to the repository.
```json
"openvlthinker-iter3": {
    "hf_hub_url": "ydeng9/OpenVLThinker-sft-iter3",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
```
* **For local data**: If using a local file, use the `"file_name"` key instead of `"hf_hub_url"`. 
```json
"my_local_dataset": {
    "file_name": "my_training_data.json",
    "formatting": "sharegpt",
    ...
}
```
