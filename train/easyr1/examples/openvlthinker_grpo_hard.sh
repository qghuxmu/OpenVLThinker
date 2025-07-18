set -x

MODEL_PATH=ydeng9/OpenVLThinker-v1.2-medium-grpo-iter3 # replace it with your local file path
SAVE_PATH=/home/user/models/OpenVLThinker-v1.2 # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=ydeng9/OpenVLThinker-grpo-hard@train \
    data.val_files=ydeng9/OpenVLThinker-grpo-hard@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=thinker_grpo_iter3_hard \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=${SAVE_PATH}
