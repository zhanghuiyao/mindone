# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import textwrap
import warnings
import numpy as np
from collections import defaultdict, deque
from collections.abc import Sized
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union

import mindspore
from mindspore import mint, nn, ops, Tensor, Parameter
from mindspore.nn.learning_rate_schedule import LearningRateSchedule

import datasets
from datasets import Dataset, IterableDataset
from packaging import version

# import from huggingface/transformers
import transformers
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerBase,
    is_wandb_available,
)
from transformers.utils import is_datasets_available, is_rich_available, ModelOutput

# import from mindone.transformers
from mindone.transformers.models.auto.modeling_auto import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from mindone.transformers.modeling_utils import MSPreTrainedModel as PreTrainedModel
from mindone.transformers.trainer import Trainer, TrainerCallback
from mindone.transformers.mindspore_adapter.data import Sampler
from mindone.transformers.mindspore_adapter.utils import enable_dynamic_shape, pynative_no_grad, NAN_TENSOR

# import from huggingface/trl
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_vllm_available
from trl.trainer.utils import (
    print_prompt_completions_sample,
)

# import from mindone.trl
from mindone.trl.models import create_reference_model
from mindone.trl.trainer.grpo_config import GRPOConfig
from mindone.trl.trainer.callbacks import SyncRefModelCallback
from mindone.trl.trainer.utils import (
    disable_dropout_in_model,
    pad,
    selective_log_softmax
)


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            if seed is not None:
                mindspore.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = np.random.permutation(self.num_samples).tolist()            
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: mindspore.Tensor) -> mindspore.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`mindspore.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `mindspore.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = ops.nanmean((tensor - ops.nanmean(tensor, keepdims=True)) ** 2)  # Compute variance ignoring NaNs
    count = mint.sum(ops.logical_not(mint.isnan(tensor)))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return mint.sqrt(variance)


def split_tensor_dict(
    tensor_dict: dict[str, Optional[mindspore.Tensor]], num_chunks: int
) -> list[dict[str, Optional[mindspore.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
        >>> x = mint.arange(12).reshape(6, 2)
        >>> y = mint.arange(6).reshape(6, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> split_tensor_dict(tensor_dict, 3)
        [
            {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
            {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
            {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
        ]
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


def shuffle_tensor_dict(tensor_dict: dict[str, Optional[mindspore.Tensor]]) -> dict[str, Optional[mindspore.Tensor]]:
    """
    Shuffles a dictionary of tensors along the first dimension in unison.

    Example:
        >>> x = mint.arange(6).reshape(3, 2)
        >>> y = mint.arange(3).reshape(3, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> shuffle_tensor_dict(tensor_dict)
        {'x': tensor([[2, 3],
                      [0, 1],
                      [4, 5]]),
         'y': tensor([[1],
                      [0],
                      [2]])}
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    batch_size = first_tensor.shape[0]
    permutation = ops.randperm(batch_size)
    return {key: tensor[permutation] if tensor is not None else None for key, tensor in tensor_dict.items()}


def nanmin(tensor: mindspore.Tensor) -> mindspore.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`mindspore.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `mindspore.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if ops.isnan(tensor).all():
        return mindspore.Tensor(float("nan"), dtype=tensor.dtype)
    return mint.min(tensor[ops.logical_not(ops.isnan(tensor))])


def nanmax(tensor: mindspore.Tensor) -> mindspore.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`mindspore.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `mindspore.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if ops.isnan(tensor).all():
        return mindspore.Tensor(np.array(float('nan')), dtype=tensor.dtype)
    return mint.max(tensor[ops.logical_not(ops.isnan(tensor))])


class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                  functions can also return None when the reward is not applicable to those samples. This is useful for
                  multi-task training where different reward functions apply to different types of samples. When a
                  reward function returns None for a sample, that reward function is excluded from the reward
                  calculation for that sample. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`]. A
            padding token, `processing_class.pad_token`, must be set. If the processing class has not set a padding
            token, `processing_class.eos_token` will be used as the default.
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[nn.optim.Optimizer], Optional[Union[LearningRateSchedule, list, float]]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, mindspore.Type) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a mindspore.Type or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(mindspore, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            raise NotImplementedError

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Cell):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(reward_funcs[i].config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = mindspore.Tensor(args.reward_weights, dtype=mindspore.float32)
        else:
            self.reward_weights = ops.ones(len(reward_funcs), dtype=mindspore.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features, *args, **kwargs):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.use_liger_loss = args.use_liger_loss
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processing_class,  #TODO: update to new_name `processing_class`
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        # TODO: add peft support
        # elif is_deepspeed_zero3_enabled() or self.is_fsdp_enabled:
        #     self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        # elif is_peft_model(model):
        #     # If PEFT is used, the reference model is not needed since the adapter can be disabled
        #     # to revert to the initial model.
        #     self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Liger loss
        if self.use_liger_loss:
            raise NotImplementedError

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = args.per_device_train_batch_size * args.steps_per_generation
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        # FIXME: Set seed with device specific
        mindspore.set_seed(args.seed)

        if self.use_vllm:
            raise NotImplementedError
        else:
            self.generation_config = GenerationConfig(
                use_cache=False,
                \
                # TODO: speedup, replace to max_new_tokens when model.generate() support dynamic shape.
                # max_new_tokens=self.max_completion_length,
                max_length=self.max_prompt_length + self.max_completion_length,
                \
                do_sample=True,
                pad_token_id=processing_class.pad_token_id,
                bos_token_id=processing_class.bos_token_id,
                eos_token_id=processing_class.eos_token_id,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
                cache_implementation=args.cache_implementation,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        # self.model.add_model_tags(self._tag_names)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = reward_func

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            
            class MSDataset:
                def __init__(self, dataset: datasets.Dataset):
                    self.dataset = dataset

                def __getitem__(self, item):
                    return self.dataset[int(item)]

                def __len__(self):
                    return len(self.dataset)
            
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
            train_dataset = MSDataset(train_dataset)
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if self.args.dataloader_prefetch_factor is not None:
            mindspore.dataset.config.set_prefetch_size(self.args.dataloader_prefetch_factor)
        
        ds_init_params = {
            "num_parallel_workers": self.args.dataloader_num_workers,
            "sampler": self._get_train_sampler(),
            "python_multiprocessing": False,
            "num_shards": getattr(self.args, "rank_size", 1),
            "shard_id": getattr(self.args, "rank", 0),
            "column_names": "item",
        }
        ds_batch_params = {
            "num_parallel_workers": self.args.dataloader_num_workers,  # num workers
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change  # per device batch size
            "per_batch_map": data_collator,  # collate function
            "drop_remainder": self.args.dataloader_drop_last,  # drop last
        }
        ds_repeat_params = {"count": 1}  # self.args.num_train_epochs            # num_train_epochs, loop at train func

        loader = mindspore.dataset.GeneratorDataset(train_dataset, **ds_init_params)
        loader = loader.batch(**ds_batch_params)
        loader = loader.repeat(**ds_repeat_params)

        print(
            f"create dataloader success, \n"
            f"\tshard_id/num_shards: {ds_init_params['shard_id']}/{ds_init_params['num_shards']}\n"
            f"\tnum_parallel_workers: {ds_init_params['num_parallel_workers']}\n"
            f"\tpython_multiprocessing: {ds_init_params['python_multiprocessing']}\n"
            f"\tper_batch_size: {ds_batch_params['batch_size']}"
        )

        return loader

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |    Accum step 0     |
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...

        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()

        if args.gradient_checkpointing_kwargs is not None:
            print(f"unuse gradient_checkpointing_kwargs: {args.gradient_checkpointing_kwargs}")

        return model

    @profiling_decorator
    def _get_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep=None):
        # last_hidden_state = unwrapped_model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # if logits_to_keep is not None:
        #     last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        # return last_hidden_state
        
        raise NotImplementedError

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None) -> mindspore.Tensor:
        batch_size = batch_size or input_ids.shape[0]  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.shape[0], batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            tuple_inputs = (input_ids_batch, attention_mask_batch)

            if self.args.enable_dynamic_shape:
                enable_dynamic_shape(model, *tuple_inputs)
            
            model_out = model(*tuple_inputs)
            if isinstance(model_out, (tuple, list)):
                logits = model_out[0]
            elif isinstance(model_out, ModelOutput):
                logits = model_out.logits
            else:
                raise ValueError
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = logits[:, -(logits_to_keep + 1):, :]

            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
            all_logps.append(logps)
        return mint.cat(all_logps, dim=0)

    def _sync_fsdp_params_to_vllm(self, module: nn.Cell, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        raise NotImplementedError

    @profiling_decorator
    def _move_model_to_vllm(self):
        raise NotImplementedError

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[mindspore.Tensor, Any]]
    ) -> dict[str, Union[mindspore.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = shuffle_tensor_dict(generation_batch)
                self._buffered_inputs = split_tensor_dict(generation_batch, self.args.steps_per_generation)
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[mindspore.Tensor, Any]]]
    ) -> dict[str, Union[mindspore.Tensor, Any]]:
        mode = "train" if self.model.training else "eval"

        # prompts = [x["prompt"] for x in inputs]
        prompts = [str(x) for x in inputs["prompt"]]

        # prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompts_text = [maybe_apply_chat_template({k: str(v[i]) for k, v in inputs.items()}, self.tokenizer)["prompt"] for i in range(len(prompts))]
        # prompt_inputs = self.processing_class(
        prompt_inputs = self.tokenizer(
            text=prompts_text, return_tensors="np", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            raise NotImplementedError
        else:
            completion_ids = self.model.generate(
                prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
            )

            # FIXME: delete it when model.generate() support dynamic shape
            completion_ids = completion_ids[:, :self.max_completion_length]

            prompt_completion_ids = mint.cat([prompt_ids, completion_ids], dim=1)

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.shape[1]
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        # is_eos = completion_ids == self.processing_class.eos_token_id
        is_eos = completion_ids == self.tokenizer.eos_token_id
        eos_idx = mint.full((is_eos.shape[0],), is_eos.shape[1], dtype=mindspore.int32)
        eos_idx[is_eos.any(axis=1)] = is_eos.int().argmax(axis=1)[is_eos.any(axis=1)]
        sequence_indices = mint.arange(is_eos.shape[1]).expand((is_eos.shape[0], -1))
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ops.logical_not(is_eos.any(axis=1))
            completion_mask = completion_mask * (ops.logical_not(truncated_completions)).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = mint.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.shape[1]  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with pynative_no_grad():
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

        # Decode the generated completions
        # completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)  # TODO: adapte to newest version
        completions_text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        inputs_example = {k: str(v[0]) for k, v in inputs.items()}
        if is_conversational(inputs_example):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = mint.zeros((len(prompts), len(self.reward_funcs)), dtype=mindspore.float32)
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Cell
                ):  # nn.Cell instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs_example):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="np", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with pynative_no_grad():
                        prev_stete = reward_func.training
                        reward_func.set_train(False)

                        reward_tuple_inputs = (reward_inputs["input_ids"], reward_inputs["attention_mask"])

                        if self.args.enable_dynamic_shape:
                            enable_dynamic_shape(reward_func, *reward_tuple_inputs)

                        # rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                        reward_out = reward_func(*reward_tuple_inputs)
                        if isinstance(reward_out, (tuple, list)):
                            logits = reward_out[0]
                        elif isinstance(reward_out, ModelOutput):
                            logits = reward_out.logits
                        else:
                            raise ValueError

                        rewards_per_func[:, i] = logits[:, 0]  # Shape (B*G,)

                        reward_func.set_train(prev_stete)

                    reward_kwargs = {}
                else:
                    # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the number
                    # of generations
                    keys = [key for key in inputs_example if key not in ["prompt", "completion", "completion_ids"]]
                    # reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    reward_kwargs = {key: [str(example) for example in inputs[key]] for key in keys}
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else NAN_TENSOR for reward in output_reward_func]

                    rewards_per_func[:, i] = mindspore.Tensor(output_reward_func, dtype=mindspore.float32)

        # If all reward functions return None for a given row, issue a detailed warning
        if ops.isnan(rewards_per_func).all(axis=1).any():
            nan_row_idx = ops.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        # rewards_per_func = all_gather(rewards_per_func)  # TODO: All-gather

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        # process_slice = slice(
        #     self.accelerator.process_index * len(prompts),
        #     (self.accelerator.process_index + 1) * len(prompts),
        # )
        # advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += attention_mask.sum().item()  #self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # log completion lengths, mean, min, max
        agg_completion_mask = completion_mask.sum(1)  #self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())

        # identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = is_eos.any(axis=1)  #self.accelerator.gather_for_metrics(is_eos.any(axis=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = mint.zeros(1)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_mask.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = ops.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(prompts_text)           #TODO: All-gather, all_gather(prompts_text)
        self._textual_logs["completion"].extend(completions_text)   #TODO: All-gather
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
        }

    def _wrap_model(self, model, dataloader=None):

        class GRPOModel(nn.Cell):
            def __init__(self, model, loss_type, temperature, epsilon_low, epsilon_high, beta, max_completion_length):
                super().__init__(auto_prefix=False)
                self.model = model
                self.loss_type = loss_type
                self.temperature = temperature
                self.epsilon_low = epsilon_low
                self.epsilon_high = epsilon_high
                self.beta = beta
                self.max_completion_length = max_completion_length

            def compute_per_token_logps(self, input_ids, attention_mask, logits_to_keep):
                batch_size = input_ids.shape[0]  # Chunk inputs into smaller batches to reduce memory peak
                all_logps = []
                for i in range(0, input_ids.shape[0], batch_size):
                    input_ids_batch = input_ids[i : i + batch_size]
                    attention_mask_batch = attention_mask[i : i + batch_size]

                    logits = self.model(
                        input_ids_batch,
                        attention_mask_batch
                    )[0]
                    
                    if logits_to_keep is not None:  # for dynamic shape
                        # slice logit
                        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                        logits = logits[:, -(logits_to_keep + 1):, :]

                        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
                        input_ids_batch = input_ids_batch[:, -logits_to_keep:]
                        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
                        # See https://github.com/huggingface/trl/issues/2770
                        logits = logits[:, -logits_to_keep:]
                    
                    else:                           # for padding branch, logits shifted right
                        _l, _r = mint.split(logits, (logits.shape[1]-1, 1), dim=1)
                        logits = mint.cat((_r, _l), dim=1)

                    # Divide logits by sampling temperature.
                    # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
                    logits = logits / self.temperature
                    logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
                    
                    all_logps.append(logps)
                return mint.cat(all_logps, dim=0)

            def construct(
                    self,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    completion_mask,
                    advantages,
                    old_per_token_logps=None,
                    ref_per_token_logps=None
            ):
                # Compute the loss
                per_token_logps = self.compute_per_token_logps(input_ids, attention_mask, logits_to_keep)

                per_token_kl = (
                    mint.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                )

                # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
                # old_per_token_logps == per_token_logps, so we can skip it's computation
                # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
                if old_per_token_logps is None:
                    old_per_token_logps = ops.stop_gradient(per_token_logps)

                coef_1 = mint.exp(per_token_logps - old_per_token_logps)
                coef_2 = mint.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
                per_token_loss1 = coef_1 * advantages.unsqueeze(1)
                per_token_loss2 = coef_2 * advantages.unsqueeze(1)
                per_token_loss = -mint.min(per_token_loss1, per_token_loss2)
                if self.beta != 0.0:
                    per_token_loss = per_token_loss + self.beta * per_token_kl

                if self.loss_type == "grpo":
                    loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
                elif self.loss_type == "bnpo":
                    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
                elif self.loss_type == "dr_grpo":
                    loss = (per_token_loss * completion_mask).sum() / (per_token_loss.shape[0] * self.max_completion_length)
                else:
                    loss = None

                return loss

        _model = GRPOModel(
            model,
            loss_type=self.loss_type,
            temperature=self.temperature,
            epsilon_low=self.epsilon_low,
            epsilon_high=self.epsilon_high,
            beta=self.beta,
            max_completion_length=self.max_completion_length
        )

        _, train_model = super()._wrap_model(_model, dataloader)

        return model, train_model

    def training_step(self, train_model, inputs):
        
        _s_time = time.time()
        inputs = self._prepare_inputs(inputs)
        print(f"prepare inputs and rewards, time cost: {(time.time()-_s_time) * 1000:.2f} ms")

        # Compute the per-token log probabilities for the model
        (
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            advantages,
            old_per_token_logps
        ) = \
            inputs["prompt_ids"], \
            inputs["prompt_mask"], \
            inputs["completion_ids"], \
            inputs["completion_mask"], \
            inputs["advantages"], \
            inputs["old_per_token_logps"]
        
        input_ids = mint.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = mint.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.shape[1]  # we only need to compute the logits for the completion tokens

        # Compute the KL divergence between the model and the reference model
        _s_time = time.time()
        if self.beta != 0.0:
            with pynative_no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep
                    )
                else:
                    # TODO: model need disable_adapter when use_peft
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, input_ids, attention_mask, logits_to_keep
                    )
        print(f"get reference logps, time cost: {(time.time()-_s_time) * 1000:.2f} ms")

        tuple_inputs = (
            input_ids,
            attention_mask,
            logits_to_keep,
            completion_mask,
            advantages,
            old_per_token_logps,
            ref_per_token_logps
        )
        
        if self.args.enable_dynamic_shape:
            # 1. FIXME: dynamic shape bug on MindSpore 2.5.0
            # enable_dynamic_shape(train_model, *tuple_inputs)

            # 2. pad to max_len (prompt_ids + completion_ids + pad_ids)
            _s_time = time.time()
            max_length = self.max_prompt_length + self.max_completion_length
            _pad_r, _pad_l = max_length - input_ids.shape[1], input_ids.shape[1] - completion_mask.shape[1]
            input_ids = ops.pad(input_ids, (0, _pad_r, 0, 0), "constant", 0)
            attention_mask = ops.pad(input_ids, (0, _pad_r, 0, 0), "constant", 0)
            logits_to_keep = None
            completion_mask = ops.pad(completion_mask, (_pad_l, _pad_r, 0, 0), "constant", 0)
            advantages = advantages
            assert old_per_token_logps is None
            ref_per_token_logps = ops.pad(ref_per_token_logps, (_pad_l, _pad_r, 0, 0), "constant", 0)
            tuple_inputs = (
                input_ids,
                attention_mask,
                logits_to_keep,
                completion_mask,
                advantages,
                old_per_token_logps,
                ref_per_token_logps
            )
            print(f"padding inputs, time cost: {(time.time()-_s_time) * 1000:.2f} ms")

        if self.use_liger_loss:
            raise NotImplementedError

        train_model.set_train()

        _s_time = time.time()
        loss, _, overflow = train_model(*tuple_inputs)
        print(f"train_model step, time cost: {(time.time()-_s_time) * 1000:.2f} ms")

        if overflow:
            print("WARNING: this train step overflow.")

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # TODO: log the metrics
        # self.log_metrics(
        #     advantages=advantages,
        #     completion_mask=completion_mask,
        #     per_token_kl=0.0,
        #     coef_1=0.0
        # )

        return loss / self.args.gradient_accumulation_steps

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        raise NotImplementedError

    def log_metrics(self, advantages, completion_mask, per_token_kl, coef_1):
        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(mean_kl.nanmean().item())
            # self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = mint.logical_and((coef_1 < 1 - self.epsilon_low), (advantages.unsqueeze(1) < 0))
        is_high_clipped = mint.logical_and((coef_1 > 1 + self.epsilon_high), (advantages.unsqueeze(1) > 0))
        is_region_clipped = mint.logical_or(is_low_clipped, is_high_clipped)

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = low_clip        #TODO: All-gather #self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = high_clip      #TODO: All-gather #self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = clip_ratio    #TODO: All-gather #self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        raise NotImplementedError

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        # transformers<=4.46
        super().log(logs)   #super().log(logs, start_time)  #TODO: adapte to newest version

        self._metrics[mode].clear()

        # if self.accelerator.is_main_process and self.log_completions:
        is_main_process = True
        if is_main_process:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._textual_logs["prompt"],
                    self._textual_logs["completion"],
                    self._textual_logs["rewards"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                    **self._textual_logs["rewards"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        raise NotImplementedError
