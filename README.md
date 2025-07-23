<div align="center">

<h1>OpenVLThinker: Complex Vision-Language Reasoning via Iterative SFT-RL Cycles</h1>

<p align="center">
  <a href="https://huggingface.co/collections/ydeng9/openvlthinker-v12-models-686f4632c23b59379c475169">🤗Models</a> • <a href="https://huggingface.co/collections/ydeng9/openvlthinker-v12-datasets-686f45e48d02e00b1585299e">🤗Data</a> • <a href="https://arxiv.org/abs/2503.17352">📄Paper</a>
</p>

</div>

We maintain our initially released model here: [Legacy model: OpenVLThinker-v1.0](https://huggingface.co/ydeng9/OpenVLThinker-7B), with our initial exploratory [blog](https://yihe-deng.notion.site/openvlthinker).

Authors: [Yihe Deng](https://yihe-deng.notion.site/yihe-deng-main), [Hritik Bansal](https://sites.google.com/view/hbansal), [Fan Yin](https://fanyin3639.github.io/), [Nanyun Peng](https://violetpeng.github.io/), [Wei Wang](https://web.cs.ucla.edu/~weiwang/), [Kai-Wei Chang](https://web.cs.ucla.edu/~kwchang/)

Our study investigates whether R1-like reasoning capabilities can be successfully integrated into large vision-language models (LVLMs) and assesses their impact on challenging multimodal reasoning tasks. We consider an approach that iteratively leverages supervised fine-tuning (SFT) on lightweight training data and Reinforcement Learning (RL) to further improve model generalization. 

As an early result, we present OpenVLThinker, a LVLM exhibiting consistently improved reasoning performance on challenging benchmarks such as MathVista, MathVerse, and MathVision.

<p align="center">
<img src="./assets/demo-vlthinker.png" width="700">
</p>

## Training

OpenVLThinker is iteratively trained in two main stages: Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL). The instructions for replicating the training process are located in their respective subdirectories.

### 1. Supervised Fine-Tuning (SFT)

This process is managed using the LLaMA-Factory framework. For complete setup and training instructions, please refer to the SFT README:
**➡️ [SFT Training Instructions](./train/llama-factory/README.md)**

### 2. Reinforcement Learning (RL)

This process is based on the EasyR1 framework. For detailed steps on running the two-stage RL training, please see the RL README:
**➡️ [RL Training Instructions](./train/easyr1/README.md)**

## Evaluation

Our model has been evaluated on several challenging benchmarks:

- Math reasoning: MathVista, MathVerse, MathVision
- General reasoning: MMMU-Pro, EMMA
- Perception: HallusionBench

<p align="center">
<img src="./assets/overview.png" width="900">
</p>

Necessary packages
```bash
pip install qwen_vl_utils
pip install mathruler
```

### Run Evaluation

We provide two evaluation scripts to handle different answer formats:

1. For OpenVLThinker evaluation:

```bash
python evaluation/eval_openvlthinker.py --dataset mathvista
```

2. For Qwen2.5-VL evaluation:

```bash
python evaluation/eval_qwen.py --dataset mathvista
```

An optional `--cuda` argument can be used to specify the GPU device (e.g., `--cuda 0`). The evaluation results, including a detailed JSON report, will be saved in the `./evaluation/outputs` directory.

### Datasets
Evaluation supports 
- `mathvista`, 
- `mathverse`, 
- `mathvision`
- EMMA (`emma-math`,`emma-chem`, `emma-code`, `emma-physics`)
- MMMU (`mmmu-pro-vision`, `mmmu-pro-4`, `mmmu-pro-10`)
- `hallusionbench`

### Special Case: MathVerse Evaluation

Due to the free-form nature of the MathVerse benchmark, we use GPT-4V to verify the model's responses. After generating the output file with the command above, run the verification script:

```bash
python evaluation/verify_mathverse_gpt4.py \
    --responses_file ./evaluation/outputs/mathverse_OpenVLThinker-v1.2.json 
```

**Note**: This requires an OPENAI_API_KEY to be set in your environment variables.

## Citation
```text
@misc{deng2025openvlthinker,
      title={OpenVLThinker: An Early Exploration to Complex Vision-Language Reasoning via Iterative Self-Improvement}, 
      author={Yihe Deng and Hritik Bansal and Fan Yin and Nanyun Peng and Wei Wang and Kai-Wei Chang},
      year={2025},
      eprint={2503.17352},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.17352}, 
}
```

## Acknowledgments

We thank [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [EasyR1](https://github.com/hiyouga/EasyR1) for open-sourcing the model training frameworks that we used in this work.