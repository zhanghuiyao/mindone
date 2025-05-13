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

import shutil
import mindspore
from dataclasses import dataclass, field
from datasets import load_dataset

# import from transformers
from mindone.transformers.models.auto.modeling_auto import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)

# import from trl
from mindone.trl import (
    GRPOConfig,
    GRPOTrainer,
)
from trl import (
    ModelConfig,
    ScriptArguments,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from mindone.transformers.mindspore_adapter import MindSporeArguments, init_environment


"""
python -i examples/scripts/grpo.py \
    --model_path Qwen/Qwen2.5-1.5B \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir ./outputs/grpo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --missing_eos_penalty 1.0
"""


@dataclass
class MyTrainingArguments(ScriptArguments, MindSporeArguments, GRPOConfig):
    model_path: str = field(default="Qwen/Qwen2.5-1.5B")
    dataset_name: str = field(default="trl-internal-testing/descriptiveness-sentiment-trl-style")
    dataset_train_split: str = field(default="descriptiveness", metadata={"help": "Dataset split to use for training."})
    output_dir: str = field(default="./outputs")
    enable_dynamic_shape: bool = field(default=True)
    enable_flash_attention: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    is_distribute: bool = field(default=False)

    bf16: bool = field(default=True)
    fp16: bool = field(default=False)


if __name__ == "__main__":
    parser = HfArgumentParser((MyTrainingArguments, ModelConfig))
    training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    init_environment(training_args)

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.model_path,
        padding_side="left"
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.model_path,
        num_labels=1,
        mindspore_dtype=mindspore.bfloat16 if training_args.bf16 else (mindspore.float16 if training_args.fp16 else None),
        use_flash_attention_2=training_args.enable_flash_attention,
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.model_path,
        mindspore_dtype=mindspore.bfloat16 if training_args.bf16 else (mindspore.float16 if training_args.fp16 else None),
        use_flash_attention_2=training_args.enable_flash_attention,
    )

    ################
    # Dataset
    ################
    dataset = load_dataset(
        training_args.dataset_name, name=training_args.dataset_config, split=training_args.dataset_train_split
    )
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    dataset_text_field = "prompt"

    # def prepare_dataset(dataset, tokenizer=None):
    #     """pre-tokenize the dataset before training; only collate during training"""
    #
    #     def tokenize(element):
    #         outputs = tokenizer(
    #             element[dataset_text_field],
    #             padding=False,
    #         )
    #         return {"input_ids": outputs["input_ids"]}
    #
    #     map_func = tokenize if tokenizer is not None else (lambda x: x)
    #     remove_columns = dataset.column_names if tokenizer is not None else None
    #
    #     return dataset.map(
    #         map_func,
    #         batched=True,
    #         remove_columns=remove_columns,
    #     )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    # with PartialState().local_main_process_first():
    # train_dataset = prepare_dataset(train_dataset, tokenizer=None)
    # eval_dataset = prepare_dataset(eval_dataset, tokenizer=None)

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=reward_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save
    trainer.save_model(training_args.output_dir)
