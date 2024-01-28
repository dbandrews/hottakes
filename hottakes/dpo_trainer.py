# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

# Note: you need to install transformers from main to run this script. See https://huggingface.co/docs/transformers/installation#install-from-source
# TODO: bump transformers version in requirements at next release.

# Taken from:
# https://github.com/huggingface/trl/blob/8e9cae8072714cea06bb39e57c692df86b6e2153/examples/scripts/dpo.py

# 0. imports
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Optional
import uuid

import torch
from datasets import Dataset, load_dataset
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, BitsAndBytesConfig

import mlflow
from trl import DPOTrainer


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    learning_rate: Optional[float] = field(default=1e-3, metadata={"help": "optimizer learning rate"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    # instrumentation
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="mlflow",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    # Custom ----
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    mlflow_experiment_name: Optional[str] = field(default=None, metadata={"help": "The name of the MLflow experiment"})
    mlflow_tracking_uri: Optional[str] = field(default=None, metadata={"help": "mlflow_tracking_uri"})
    mlflow_run_name: Optional[str] = field(default=None, metadata={"help": "Name of the MLflow run"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    eval_steps: Optional[int] = field(default=500, metadata={"help": "Number of updates steps before eval"})
    chosen_pcu_threshold: Optional[int] = field(
        default=1000, metadata={"help": "The minimum number of upvotes on the top comment"}
    )
    rejected_pcd_threshold: Optional[int] = field(
        default=1000, metadata={"help": "The minimum number of downvotes on the worst comment"}
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. load a pretrained model
    # Step 1: Load the model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        # use_flash_attention_2=True,
    )
    model.enable_input_require_grads()

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    # tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 1b: Define the LoraConfig
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    # # Step 2: Load the dataset
    dataset = load_dataset("json", data_files=script_args.dataset_name, split="train")
    # Step 2.1: Filter the dataset based on pcu and pcd
    print(f"Dataset size before filtering: {len(dataset)}")
    dataset = dataset.filter(
        lambda x: x["chosen_pcu"] > script_args.chosen_pcu_threshold
        and x["rejected_pcd"] > script_args.rejected_pcd_threshold
    )
    print(f"Dataset size after filtering: {len(dataset)}")
    # Convert the dataset to the format expected by the DPOTrainer, with 3 keys and lists under each
    # key. The keys are "prompt", "chosen", and "rejected". The values under each key are lists of
    # strings.
    dataset = dataset.map(
        lambda sample: {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }
    )
    # Create valid huggingface dataset out of random fraction of train
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]
    # train_dataset = get_hh("train", sanity_check=script_args.sanity_check)
    # valid_dataset = get_hh("test", sanity_check=script_args.sanity_check)
    print("Dataset formatted")

    # 4. initialize training arguments:
    mlflow_run_name = f"{script_args.mlflow_run_name}-dpo-{uuid.uuid4()}"
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        max_steps=script_args.max_steps,
        remove_unused_columns=False,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=10,  # match results in blog post
        eval_steps=script_args.eval_steps,
        save_steps=script_args.save_steps,
        output_dir=Path(script_args.output_dir) / mlflow_run_name,
        # optim="rmsprop",
        warmup_steps=150,
        report_to=script_args.report_to,
        # bf16=True,
        gradient_checkpointing=script_args.gradient_checkpointing,
        run_name=mlflow_run_name,
        # TODO: uncomment that on the next transformers release
        # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
    )

    # 5. initialize the DPO trainer
    # Step 5: Define the Trainer
    os.environ["MLFLOW_EXPERIMENT_NAME"] = script_args.mlflow_experiment_name
    os.environ["MLFLOW_TRACKING_URI"] = script_args.mlflow_tracking_uri
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        # max_target_length=script_args.max_target_length,
        max_prompt_length=script_args.max_prompt_length,
        # generate_during_eval=True,
        peft_config=peft_config,
    )

    # 6. train
    with mlflow.start_run(run_name=mlflow_run_name):
        for arg in ["chosen_pcu_threshold", "rejected_pcd_threshold"]:
            mlflow.log_param(arg, getattr(script_args, arg))

        dpo_trainer.train()
