# Usage: deepspeed train_lora.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>

# Adapted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json
import safetensors
from dataclasses import dataclass, field
import logging
import pathlib
import typing
import os
from datasets import load_dataset

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
import transformers
from transformers import Trainer, BitsAndBytesConfig
from transformers.integrations import deepspeed
import torch

from train import (
    DataArguments,
    ModelArguments,
    rank0_print
)


# os.environ["CUDA_VISIBLE_DEVICES"] = 0

FULL_PROMPT_TEMPLATE = "{input}{output}"
NO_OUTPUT_PROMPT_TEMPLATE = "{input}"

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: typing.Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    flash_attn: bool = False
    resume_from_checkpoint: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_A_dir: str = ""


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def load_lora_A_weights(model, lora_A_dir, target_modules=["v_proj"], output_dir=None):
    """
    Load lora_A parameters from a pretrained adapter directory.
    
    Args:
        model: The PEFT model to load weights into
        lora_A_dir: Directory containing the pretrained adapter weights
        target_modules: List of module names to load lora_A from (default: ["v_proj"])
    """
    adapters_weights = None
    extensions = ["safetensors", "bin"]
    
    for extension in extensions:
        lora_A_path = os.path.join(lora_A_dir, f"adapter_model.{extension}")
        if os.path.exists(lora_A_path):
            print(f"Checking lora_A parameters from {', '.join(target_modules)} from {lora_A_dir}")
            if extension == "safetensors":
                adapters_weights = safetensors.torch.load_file(lora_A_path, device="cpu")
            else:
                adapters_weights = torch.load(lora_A_path, map_location="cpu")
            print(f"Found adapter weights from {lora_A_path}")
            break

    if adapters_weights is not None:
        lora_A_filtered = {
            k: v for k, v in adapters_weights.items()
            if "lora_A" in k and any(module in k for module in target_modules)
        }
        if len(lora_A_filtered) > 0:
            set_peft_model_state_dict(model, lora_A_filtered, adapter_name="default")
            print(f"Successfully loaded {len(lora_A_filtered)} lora_A parameters from {', '.join(target_modules)}")
            
            # Freeze the loaded lora_A parameters (make them non-trainable)
            frozen_count = 0
            for name, param in model.named_parameters():
                if "lora_A" in name and any(module in name for module in target_modules):
                    param.requires_grad = False
                    frozen_count += 1
            print(f"Frozen {frozen_count} lora_A parameters from {', '.join(target_modules)} (not trainable)")
        else:
            print(f"Warning: No lora_A parameters found for {', '.join(target_modules)} in adapter weights")
    else:
        print(f"Warning: No adapter files found at {lora_A_dir}")
        if "plan" in output_dir:
            output_dir = output_dir.replace("v3", "v2")
        print(f"Will be saved at: {output_dir}")

    return model, output_dir
    

def load_and_process_dataset(tokenizer, data_args):
    print("Loading data...")

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < tokenizer.model_max_length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = FULL_PROMPT_TEMPLATE.format(
            input=data_point["input"],
            output=data_point["output"]
        )
        tokenized_full_prompt = tokenize(full_prompt)
        user_prompt = NO_OUTPUT_PROMPT_TEMPLATE.format(input=data_point["input"])
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]
        return tokenized_full_prompt

    train_data = load_dataset("json", data_files=data_args.data_path)
    if data_args.eval_data_path:
        eval_data = load_dataset("json", data_files=data_args.data_path)
        eval_data = eval_data["train"].shuffle().map(generate_and_tokenize_prompt)
    else:
        eval_data = None
    train_data = train_data["train"].shuffle().map(generate_and_tokenize_prompt)
    return train_data, eval_data




def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP and ZeRO3 are both currently incompatible with QLoRA."
            )

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if lora_args.q_lora
        else None,
        torch_dtype=compute_dtype
    )
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )
        if not ddp and torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            model.is_parallelizable = True
            model.model_parallel = True

    model = get_peft_model(model, lora_config)
    model, training_args.output_dir = load_lora_A_weights(model, lora_args.lora_A_dir, target_modules=["v_proj"], output_dir=training_args.output_dir)
    
    if training_args.flash_attn:
        for name, module in model.named_modules():
            if "norm" in name:
                module = module.to(compute_dtype)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    module = module.to(compute_dtype)
    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
    )

    if "llama" in model_args.model_name_or_path.lower():
        model.config.pad_token_id = tokenizer.pad_token_id = 128009  # same as eos token id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
    if "mistral" in model_args.model_name_or_path.lower():
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # same as unk token id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
    if "qwen" in model_args.model_name_or_path.lower():
        model.config.pad_token_id = tokenizer.pad_token_id = 151643
        model.config.bos_token_id = tokenizer.bos_token_id = 151643  # Qwen2.5 bos token id
        model.config.eos_token_id = tokenizer.eos_token_id = 151645

    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    train_data, eval_data = load_and_process_dataset(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        # **data_module
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False

    print("all_right")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")) and training_args.resume_from_checkpoint :
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        # use deepspeed engine internal function to gather state dict
        # state_dict_zero3 contains whole parameters of base and lora adapters
        # we will not extract lora parameters since peft save_pretrained will do that
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
        state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if training_args.local_rank == 0:
            state_dict = state_dict_zero3
    else:
        # in other mode we use original code from fastchat team, to make sure our change is minimum
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), lora_args.lora_bias
        )

    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)


if __name__ == "__main__":
    train()
