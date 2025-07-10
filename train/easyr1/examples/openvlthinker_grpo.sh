set -x

MODEL_PATH=ydeng9/openvlthinker_grpo_iter3_medium # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=ydeng9/hard_better_7k@train \
    data.val_files=ydeng9/hard_better_7k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=thinker_grpo_iter3_hard \
    trainer.n_gpus_per_node=8
