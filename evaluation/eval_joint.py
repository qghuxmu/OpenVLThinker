import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
import json
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO
import argparse
from mathruler.grader import grade_answer, extract_boxed_content
import re
from pathlib import Path
from enum import Enum


import copy
import inspect
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
import torch.distributed as dist
from huggingface_hub import file_exists
from packaging import version
from torch import nn

from transformers.cache_utils import (
    Cache,
    DynamicCache,
    EncoderDecoderCache,
    QuantizedCache,
    StaticCache,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.dynamic_module_utils import (
    check_python_requirements,
    get_cached_module_file,
    get_class_in_module,
    resolve_trust_remote_code,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.masking_utils import create_masks_for_generate
from transformers.pytorch_utils import isin_mps_friendly
from transformers.tokenization_utils import ExtensionsTrie
from transformers.utils import (
    ModelOutput,
    is_accelerate_available,
    is_hqq_available,
    is_optimum_quanto_available,
    is_torchdynamo_exporting,
    # logging,
)
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.candidate_generator import (
    AssistantVocabTranslatorCache,
    AssistedCandidateGenerator,
    AssistedCandidateGeneratorDifferentTokenizers,
    CandidateGenerator,
    EarlyExitCandidateGenerator,
    PromptLookupCandidateGenerator,
    UniversalSpeculativeDecodingGenerator,
    _prepare_attention_mask,
    _prepare_token_type_ids,
)
from transformers.generation.configuration_utils import (
    ALL_STATIC_CACHE_IMPLEMENTATIONS,
    DEPRECATED_STATIC_CACHE_IMPLEMENTATIONS,
    STATIC_CACHE_IMPLEMENTATIONS,
    GenerationConfig,
    GenerationMode,
)
from transformers.generation.continuous_batching import ContinuousMixin
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    MinPLogitsWarper,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from transformers.generation.stopping_criteria import (
    ConfidenceCriteria,
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)
from transformers.generation.utils import (
    GenerateOutput,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetType(Enum):
    MATHVISTA = "mathvista"
    MATHVERSE = "mathverse"
    MATHVISION = "mathvision"
    SFTSEED = "sftseed"
    HALLUSIONBENCH = "hallusionbench"
    EMMA_MATH = "emma-math"
    EMMA_CHEM = "emma-chem"
    EMMA_CODE = "emma-code"
    EMMA_PHYSICS = "emma-physics"
    MMMU_PRO_10 = "mmmu-pro-10"
    MMMU_PRO_4 = "mmmu-pro-4"
    MMMU_PRO_VISION = "mmmu-pro-vision"

@dataclass
class DatasetConfig:
    name: str
    split: str
    image_field: str
    response_field: str
    caption_field: Optional[str] = None
    instruction_field: Optional[str] = None
    subset: Optional[str] = None
    choices_field: Optional[str] = None
    options_field: Optional[str] = None
    source_field: Optional[str] = None

@dataclass
class ModelConfig:
    model_name: str
    text_model_name: str
    max_new_tokens: int = 2048
    top_p: float = 0.001
    top_k: int = 1
    temperature: float = 0.01
    repetition_penalty: float = 1.0
    alpha: float = 0.5

class ImageProcessor:
    def __init__(self, model_config: ModelConfig, device: str, text_device: str):
        self.device = device
        self.text_device = text_device
        self.model_config = model_config
        self.model = self._load_model()
        self.processor = self._load_processor()

        # Load text-only model
        logger.info(f"Loading text-only model {model_config.text_model_name}")
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_config.text_model_name)
        self.text_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.text_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=self.text_device
        )

    def _load_model(self) -> Qwen2_5_VLForConditionalGeneration:
        try:
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_config.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=self.device
            )
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _load_processor(self) -> AutoProcessor:
        try:
            return AutoProcessor.from_pretrained(self.model_config.model_name)
        except Exception as e:
            logger.error(f"Failed to load processor: {str(e)}")
            raise

    def generate_answer(self, image_url: str, image_caption: str, instruction: str) -> Optional[str]:
        # encode vlm messages
        vlm_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": instruction + "\n\nYou FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."},
                ],
            }
        ]
        vlm_text = self.processor.apply_chat_template(
            vlm_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(vlm_messages)
        vlm_inputs = self.processor(
            text=[vlm_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # encode llm messages
        llm_messages = [
            {
                "role": "user",
                "content": "Image Description: " + image_caption + "\n\n" + instruction + "\n\nYou FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
            }
        ]
        llm_text = self.text_tokenizer.apply_chat_template(
            llm_messages, tokenize=False, add_generation_prompt=True
        )
        llm_inputs = self.text_tokenizer(
            text=[llm_text],
            padding=True,
            return_tensors="pt",
        ).to(self.text_device)


        # prepare vlm inputs for generation
        vlm_input_ids, vlm_model_kwargs = prepare_for_generation(
            self.model,
            **vlm_inputs,
            do_sample=True,
            max_new_tokens=self.model_config.max_new_tokens,
            top_p=self.model_config.top_p,
            top_k=self.model_config.top_k,
            temperature=self.model_config.temperature,
            repetition_penalty=self.model_config.repetition_penalty,
        )

        # prepare llm inputs for generation
        llm_input_ids, llm_model_kwargs = prepare_for_generation(
            self.text_model,
            **llm_inputs,
            do_sample=True,
            max_new_tokens=self.model_config.max_new_tokens,
            top_p=self.model_config.top_p,
            top_k=self.model_config.top_k,
            temperature=self.model_config.temperature,
            repetition_penalty=self.model_config.repetition_penalty,
        )

        llm_prompt_length = llm_input_ids.shape[1]

        for _ in range(self.model_config.max_new_tokens):
            # vlm forward
            vlm_inputs = self.model.prepare_inputs_for_generation(vlm_input_ids, **vlm_model_kwargs)
            vlm_outputs = self.model(**vlm_inputs, return_dict=True)
            vlm_model_kwargs = self.model._update_model_kwargs_for_generation(
                vlm_outputs,
                vlm_model_kwargs,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
            )

            # text-only forward
            llm_inputs = self.text_model.prepare_inputs_for_generation(llm_input_ids, **llm_model_kwargs)
            llm_outputs = self.text_model(**llm_inputs, return_dict=True)
            llm_model_kwargs = self.text_model._update_model_kwargs_for_generation(
                llm_outputs,
                llm_model_kwargs,
                is_encoder_decoder=self.text_model.config.is_encoder_decoder,
            )

            # merge logits
            vlm_next_token_logits = vlm_outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=vlm_input_ids.device)
            llm_next_token_logits = llm_outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=vlm_input_ids.device)
            next_token_logits = (1 - self.model_config.alpha) * vlm_next_token_logits + self.model_config.alpha * llm_next_token_logits
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            # append token
            vlm_input_ids = torch.cat([vlm_input_ids, next_tokens[:, None]], dim=-1)
            llm_input_ids = torch.cat([llm_input_ids, next_tokens[:, None].to(self.text_device)], dim=-1)

            if next_tokens.item() == self.text_tokenizer.eos_token_id:
                break

            del vlm_outputs, llm_outputs


        llm_generated_ids_trimmed = [
                out_ids[llm_prompt_length:] for out_ids in llm_input_ids
            ]

        print(self.processor.batch_decode(
            llm_generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0])
        return self.processor.batch_decode(
            llm_generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        


@torch.no_grad()
def prepare_for_generation(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    use_model_defaults: Optional[bool] = None,
    custom_generate: Optional[Union[str, Callable]] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    r"""

    Generates sequences of token ids for models with a language modeling head.

    <Tip warning={true}>

    Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    model's default generation configuration. You can override any `generation_config` by passing the corresponding
    parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    For an overview of generation strategies and code examples, check out the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        generation_config ([`~generation.GenerationConfig`], *optional*):
            The generation configuration to be used as base parametrization for the generation call. `**kwargs`
            passed to generate matching the attributes of `generation_config` will override them. If
            `generation_config` is not provided, the default will be used, which has the following loading
            priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
            configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
            default values, whose documentation should be checked to parameterize generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            Custom logits processors that complement the default logits processors built from arguments and
            generation config. If a logit processor is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            Custom stopping criteria that complements the default stopping criteria built from arguments and a
            generation config. If a stopping criteria is passed that is already created with the arguments or a
            generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
            sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
            intended for advanced users.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], list[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://huggingface.co/papers/2010.00904).
        synced_gpus (`bool`, *optional*):
            Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
            to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
            deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
        assistant_model (`PreTrainedModel`, *optional*):
            An assistant model that can be used to accelerate generation. The assistant model must have the exact
            same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
            is much faster than running generation with the model you're calling generate from. As such, the
            assistant model should be much smaller.
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The negative prompt needed for some processors such as CFG. The batch size must match the input batch
            size. This is an experimental feature, subject to breaking API changes in future versions.
        negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention_mask for `negative_prompt_ids`.
        use_model_defaults (`bool`, *optional*):
            When it is `True`, unset parameters in `generation_config` will be set to the model-specific default
            generation configuration (`model.generation_config`), as opposed to the global defaults
            (`GenerationConfig()`). If unset, models saved starting from `v4.50` will consider this flag to be
            `True`.
        custom_generate (`str` or `Callable`, *optional*):
            One of the following:
            - `str` (Hugging Face Hub repository name): runs the custom `generate` function defined at
                `custom_generate/generate.py` in that repository instead of the standard `generate` method. The
                repository fully replaces the generation logic, and the return type may differ.
            - `str` (local repository path): same as above but from a local path, `trust_remote_code` not required.
            - `Callable`: `generate` will perform the usual input preparation steps, then call the provided callable to
                run the decoding loop.
            For more information, see [the docs](../../generation_strategies#custom-generation-methods).
        kwargs (`dict[str, Any]`, *optional*):
            Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
            specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateDecoderOnlyOutput`],
                - [`~generation.GenerateBeamDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
    """
    # 0. If requested, load an arbitrary generation recipe from the Hub and run it instead
    trust_remote_code = kwargs.pop("trust_remote_code", None)
    if custom_generate is not None and isinstance(custom_generate, str):
        # Get all `generate` arguments in a single variable. Custom functions are responsible for handling them:
        # they receive the same inputs as `generate`, with `model` instead of `self` and excluding the arguments to
        # trigger the custom generation. They can access to methods from `GenerationMixin` through `model`.
        global_keys_to_exclude = {
            "self",
            "kwargs",
            "global_keys_to_exclude",
            "trust_remote_code",
            "custom_generate",
        }
        generate_arguments = {key: value for key, value in locals().items() if key not in global_keys_to_exclude}
        generate_arguments.update(kwargs)

        custom_generate_function = self.load_custom_generate(
            custom_generate, trust_remote_code=trust_remote_code, **kwargs
        )
        return custom_generate_function(model=self, **generate_arguments)

    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
    assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation

    generation_config, model_kwargs = self._prepare_generation_config(
        generation_config, use_model_defaults, **kwargs
    )
    self._validate_model_kwargs(model_kwargs.copy())
    self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

    # 2. Set generation parameters if not already defined
    if synced_gpus is None:
        synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    device = inputs_tensor.device
    self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    # decoder-only models must use left-padding for batched generation.
    if not self.config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config._pad_token_tensor is not None
            and batch_size > 1
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    # 4. Define other model kwargs
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        generation_config.use_cache = True

    if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config, model_kwargs
        )
    elif kwargs_has_attention_mask:
        # TODO (joao): generalize this check with other types of inputs
        if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
            raise ValueError("`attention_mask` passed to `generate` must be 2D.")

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    # Expand inputs depending on the generation mode
    input_ids, model_kwargs = self._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=max(generation_config.num_beams, generation_config.num_return_sequences),
        is_encoder_decoder=self.config.is_encoder_decoder,
        **model_kwargs,
    )

    if generation_config.token_healing:
        input_ids = self.heal_tokens(input_ids, tokenizer)

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = self._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
    # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
    # dynamically overrides this value as it can need more than the last token logits
    if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
        model_kwargs["logits_to_keep"] = 1

    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. Prepare the cache.
    # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
    # - different models have a different cache name expected by the model (default = "past_key_values")
    # - `max_length`, prepared above, is used to determine the maximum cache length
    max_cache_length = generation_config.max_length - 1
    if (
        inputs_tensor.shape[1] != input_ids_length
        and model_input_name == "inputs_embeds"
        and not self.config.is_encoder_decoder
    ):
        max_cache_length += inputs_tensor.shape[1]
    self._prepare_cache_for_generation(
        generation_config, model_kwargs, assistant_model, batch_size, max_cache_length
    )

    # 8. determine generation mode
    generation_mode = generation_config.get_generation_mode(assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 9. prepare logits processors and stopping criteria
    prepared_logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )
    prepared_stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
    )

    # Set model_kwargs `use_cache` so we can use it later in forward runs
    model_kwargs["use_cache"] = generation_config.use_cache

    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    model_forward = self.__call__
    compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        # If we use FA2 and a static cache, we cannot compile with fullgraph
        if self.config._attn_implementation == "flash_attention_2":
            # only raise warning if the user passed an explicit compile-config
            if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                logger.warning_once(
                    "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                    "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                )
                generation_config.compile_config.fullgraph = False
        model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True

    return input_ids, model_kwargs


def get_dataset_config(dataset_type: DatasetType) -> DatasetConfig:
    configs = {
        DatasetType.MATHVISTA: DatasetConfig(
            name="limazhiluyao/MathVista-caption",
            split="testmini",
            image_field="decoded_image",
            caption_field="caption",
            instruction_field="query",
            response_field="answer",
            choices_field="choices"
        ),
        DatasetType.MATHVERSE: DatasetConfig(
            name="AI4Math/MathVerse",
            subset="testmini",
            split="testmini",
            image_field="image",
            instruction_field="query_cot",
            response_field="answer"
        ),
        DatasetType.MATHVISION: DatasetConfig(
            name="MathLLMs/MathVision",
            split="testmini",
            image_field="decoded_image",
            instruction_field="question",
            response_field="answer",
            options_field="options"
        ),
        DatasetType.SFTSEED: DatasetConfig(
            name="ydeng9/sft_seed",
            split="train",
            image_field="decoded_image",
            instruction_field="problem",
            response_field="answer",
            source_field="source"
        ),
        DatasetType.HALLUSIONBENCH: DatasetConfig(
            name="lmms-lab/HallusionBench",
            split="image",
            image_field="image",
            instruction_field="question",
            response_field="gt_answer"
        ),
        DatasetType.EMMA_MATH: DatasetConfig(
            name="luckychao/EMMA",
            subset="Math",
            split="test",
            image_field="image_1",
            instruction_field="question",
            response_field="answer",
            options_field="options"
        ),
        DatasetType.EMMA_CHEM: DatasetConfig(
            name="luckychao/EMMA",
            subset="Chemistry",
            split="test",
            image_field=["image_1","image_2","image_3","image_4","image_5"],
            instruction_field="question",
            response_field="answer",
            options_field="options"
        ),
        DatasetType.EMMA_CODE: DatasetConfig(
            name="luckychao/EMMA",
            subset="Coding",
            split="test",
            image_field=["image_1","image_2","image_3","image_4","image_5"],
            instruction_field="question",
            response_field="answer",
            options_field="options"
        ),
        DatasetType.EMMA_PHYSICS: DatasetConfig(
            name="luckychao/EMMA",
            subset="Physics",
            split="test",
            image_field=["image_1","image_2","image_3","image_4","image_5"],
            instruction_field="question",
            response_field="answer",
            options_field="options"
        ),
        DatasetType.MMMU_PRO_10: DatasetConfig(
            name="MMMU/MMMU_Pro",
            subset="standard (10 options)",
            split="test",
            image_field=["image_1","image_2","image_3","image_4","image_5","image_6","image_7"],
            instruction_field="question",
            response_field="answer",
            options_field="options"
        ),
        DatasetType.MMMU_PRO_4: DatasetConfig(
            name="MMMU/MMMU_Pro",
            subset="standard (4 options)",
            split="test",
            image_field=["image_1","image_2","image_3","image_4","image_5","image_6","image_7"],
            instruction_field="question",
            response_field="answer",
            options_field="options"
        ),
        DatasetType.MMMU_PRO_VISION: DatasetConfig(
            name="MMMU/MMMU_Pro",
            subset="vision",
            split="test",
            image_field="image",
            response_field="answer",
            options_field="options"
        ),
    }
    return configs[dataset_type]

def load_image_dataset(dataset_config: DatasetConfig) -> List[Dict]:
    """
    Load dataset from Hugging Face and extract image URLs and metadata
    """
    try:
        if dataset_config.subset:
            data = load_dataset(dataset_config.name, dataset_config.subset, split=dataset_config.split)
        else:
            data = load_dataset(dataset_config.name, split=dataset_config.split)
        items = []
        for item in data:
            if isinstance(dataset_config.image_field, list):
                dataset_item = {
                    'image_url': [item.get(x) for x in dataset_config.image_field if item.get(x) is not None],
                    'instruction': item.get(dataset_config.instruction_field, ''),
                    'response': item.get(dataset_config.response_field, ''),
                }
            else:
                dataset_item = {
                    'image_url': item[dataset_config.image_field],
                    'image_caption': item.get(dataset_config.caption_field, ''),
                    'instruction': item.get(dataset_config.instruction_field, ''),
                    'response': item.get(dataset_config.response_field, ''),
                }
            if dataset_config.choices_field:
                dataset_item['choices'] = item.get(dataset_config.choices_field)
            if dataset_config.options_field:
                dataset_item['options'] = item.get(dataset_config.options_field, [])
            if dataset_config.source_field:
                dataset_item['source'] = item.get(dataset_config.source_field, '')
            items.append(dataset_item)
        return items
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

def save_descriptions(descriptions: List[Dict], output_file: str) -> None:
    """
    Save generated descriptions to a JSON file
    """
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(descriptions, f, indent=2)
        logger.info(f"Saved {len(descriptions)} descriptions to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save descriptions: {str(e)}")
        raise

def process_response(response: str, choices: Optional[List[str]], options: Optional[List[str]] = None) -> str:
    if choices is not None:
        try:
            response_index = choices.index(response)
            return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][response_index]
        except ValueError:
            pass
    if options is not None and len(options) > 0:
        try:
            response_index = options.index(response)
            return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][response_index]
        except ValueError:
            pass
    return response

def format_instruction(instruction: str, options: Optional[List[str]] = None, yes_no: bool = False, vision: bool = False) -> str:
    options = eval(options) if isinstance(options, str) else options
    if vision:
        prompt_hint = "Hint: Please answer the question shown in the image."
        if options and len(options) > 0:
            prompt_hint += " Provide the correct option letter, e.g., A, B, C, D, E, at the end."
            choice_list = "\n".join(f"({chr(65+i)}) {opt}" for i, opt in enumerate(options))
            return f"{prompt_hint}\nChoices:\n{choice_list}"
        return prompt_hint
    elif yes_no:
        prompt_hint = "Hint: Please answer the question requiring an answer of yes or no."
        return f"{prompt_hint}\nQuestion: {instruction}"
    elif options and len(options) > 0:
        prompt_hint = "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, E, at the end."
        choice_list = "\n".join(f"({chr(65+i)}) {opt}" for i, opt in enumerate(options))
        return f"{prompt_hint}\nQuestion: {instruction}\nChoices:\n{choice_list}"
    else:
        prompt_hint = "Hint: Please answer the question requiring an answer."
        return f"{prompt_hint}\nQuestion: {instruction}"

def main():
    parser = argparse.ArgumentParser(description='Evaluate Qwen model on various math datasets')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device number to use')
    parser.add_argument('--text_cuda', type=int, default=1, help='CUDA device number to use')
    parser.add_argument('--model_path', type=str, help='Path to the model', 
                      default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument('--text_model_path', type=str, help='Path to the text-only model', 
                      default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument('--dataset', type=str, choices=['mathvista', 'mathverse', 'mathvision', 'sftseed', 'hallusionbench', 'emma-math', 'emma-chem', 'emma-code', 'emma-physics', 'mmmu-pro-10', 'mmmu-pro-4', 'mmmu-pro-vision'],
                      default='mathvista', help='Dataset to evaluate on')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory', 
                      default="./evaluation/outputs")
    args = parser.parse_args()
    
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    text_device = f"cuda:{args.text_cuda}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Using device for text model: {text_device}")

    # Configuration
    dataset_type = DatasetType(args.dataset)
    dataset_config = get_dataset_config(dataset_type)
    model_config = ModelConfig(
        model_name=args.model_path,
        text_model_name=args.text_model_path
    )
    
    output_file = f"{args.output_dir}/{dataset_type.value}_{model_config.model_name.split('/')[-1]}.json"
    
    # Initialize processor and model
    logger.info(f"Loading model {model_config.model_name}")
    processor = ImageProcessor(model_config, device, text_device)
    
    # Load dataset
    logger.info(f"Loading dataset {dataset_config.name}")
    data = load_image_dataset(dataset_config)
    
    descriptions = []
    correct = 0
    
    # For SFTSEED dataset, track accuracy per source
    source_correct = {}
    source_total = {}

    # Process each image
    for i, item in tqdm(enumerate(data), total=len(data), desc="Processing images"):
        correct_flag = 0
        if dataset_type == DatasetType.MATHVISION or dataset_type == DatasetType.EMMA_MATH or dataset_type == DatasetType.EMMA_CHEM or dataset_type == DatasetType.EMMA_CODE or dataset_type == DatasetType.EMMA_PHYSICS or dataset_type == DatasetType.MMMU_PRO_10 or dataset_type == DatasetType.MMMU_PRO_4:
            formatted_instruction = format_instruction(item['instruction'], item.get('options'))
        elif dataset_type == DatasetType.HALLUSIONBENCH:
            formatted_instruction = format_instruction(item['instruction'], yes_no=True)
        elif dataset_type == DatasetType.MMMU_PRO_VISION:
            formatted_instruction = format_instruction(item['instruction'], item.get('options'), vision=True)
        else:
            formatted_instruction = item['instruction']
        answer = processor.generate_answer(item['image_url'], item['image_caption'], formatted_instruction)
        reasoning = answer

        if answer:
            answer = extract_boxed_content(answer)
            direct_answer = reasoning.lower().split("Answer:")[-1].strip()

            if dataset_type == DatasetType.MMMU_PRO_10 or dataset_type == DatasetType.MMMU_PRO_4 or dataset_type == DatasetType.MMMU_PRO_VISION:
                processed_response = item['response']
            else:
                processed_response = process_response(
                    item['response'],
                    item.get('choices'),
                    item.get('options')
                )
            if dataset_type == DatasetType.HALLUSIONBENCH:
                processed_response = "Yes" if processed_response == "1" else "No"
            
            if processed_response.lower() == answer.lower() or grade_answer(processed_response, answer) or \
                processed_response.lower() == direct_answer.lower() or grade_answer(processed_response, direct_answer):
                correct += 1
                correct_flag = 1

                if dataset_type == DatasetType.SFTSEED and 'source' in item:
                    source = item['source']
                    if source not in source_correct:
                        source_correct[source] = 0
                        source_total[source] = 0
                    source_correct[source] += 1
        else:
            answer = "Failed to generate."
            logger.warning(f"Failed to generate answer for question {i}")

        if dataset_type == DatasetType.SFTSEED and 'source' in item:
            source = item['source']
            if source not in source_total:
                source_total[source] = 0
            source_total[source] += 1

        description = {
            'instruction': item['instruction'],
            'response': item['response'],
            'reasoning': reasoning,
            'answer': answer,
            'correct': correct_flag
        }
        
        if dataset_type == DatasetType.SFTSEED and 'source' in item:
            description['source'] = item['source']
            
        descriptions.append(description)
        
        # Save periodically
        if (i + 1) % 10 == 0:
            save_descriptions(descriptions, output_file)
    
    # Final save
    save_descriptions(descriptions, output_file)
    accuracy = correct / len(data)
    logger.info(f"Completed! Final accuracy: {accuracy:.4f}")

    if dataset_type == DatasetType.SFTSEED and source_correct:
        logger.info("Accuracy per source:")
        for source in sorted(source_correct.keys()):
            source_accuracy = source_correct[source] / source_total[source]
            logger.info(f"  {source}: {source_accuracy:.4f} ({source_correct[source]}/{source_total[source]})")

if __name__ == "__main__":
    main() 
