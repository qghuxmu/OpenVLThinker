# RL (GRPO)

This readme outlines the process for SFT. The code is sourced from [**EasyR1**](https://github.com/hiyouga/EasyR1).

## ⚙️ Setup

1.  **Create and activate the conda environment:**
    ```bash
    conda create -n vl-rl python==3.10
    conda activate vl-rl
    ```

2.  **Install dependencies:**
    ```bash
    pip install -e .
    ```

3.  **(Optional) Login to Weights & Biases:**
    To enable experiment tracking, log in to your `wandb` account.
    ```bash
    wandb login
    ```

## Training Script
To replicate the training process, first train the intermediate checkpoint and then use that checkpoint to train the final model. 

* **Intermediate Model**: Run the following script to train the intermediate checkpoint.
    ```bash
    bash examples/openvlthinker_grpo_medium.sh
    ```

* **Final Model**: After training the intermediate model, run this script to train the final `OpenVLThinker-7B-v1.2` model.
    ```bash
    bash examples/openvlthinker_grpo_hard.sh
    ```

### Convert to Hugging Face Format

Once training is complete, the saved checkpoints need to be converted into the standard Hugging Face format for use.

Use the `model_merger.py` script:

```bash
python scripts/model_merger.py --local_dir /path/to/your/saved/checkpoints/actor
```

Make sure to replace /path/to/your/saved/checkpoints/actor with the actual directory where your trained actor model checkpoints are stored.