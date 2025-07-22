set -x

MODEL_PATH=ydeng9/OpenVLThinker-v1.2-sft-iter3 # replace it with your local file path
SAVE_PATH=/home/user/models/OpenVLThinker-v1.2-medium-grpo-iter3 # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=ydeng9/medium_grpo_only@train \
    data.val_files=ydeng9/medium_grpo_only@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.freeze_vision_tower=True \
    trainer.experiment_name=thinker_grpo_iter3_medium \
    trainer.n_gpus_per_node=8 \
    trainer.total_episodes=5 \
    trainer.save_checkpoint_path=${SAVE_PATH} 
