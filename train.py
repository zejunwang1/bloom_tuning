# coding=utf-8
# reference link: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

import torch
import transformers
from datasets import load_dataset
from dataclasses import dataclass, field
from transformers import (
    BloomForCausalLM, 
    BloomTokenizerFast,
    HfArgumentParser,
    set_seed,
    Trainer
)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "HuggingFace model name or path."})


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to the training data."})
    eval_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})
    test_size: int = field(default=0, metadata={"help": "Size of test dataset."})
    max_input_length: int = field(default=128, metadata={"help": "Maximum length of input."})
    max_output_length: int = field(default=512, metadata={"help": "Maximum length of output."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str = field(default=None)
    optim: str = field(default="adamw_torch")


@dataclass
class DataCollator:
    """Collate examples for supervised fine-tuning."""
    
    pad_token_id: int = 0
    
    def __call__(self, instances):
        input_ids, labels = tuple([torch.tensor(instance[key]) for instance in instances] 
                            for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.pad_token_id)
        )


def build_dataset(data_or_path, tokenizer, max_input_length=128, max_output_length=512):
    """Dataset for supervised fine-tuning."""
    data = data_or_path
    if isinstance(data_or_path, str):
        data = load_dataset(path="json", data_files=data_or_path)["train"]
    column_names = data.column_names
        
    def tokenize_function(example):
        question = example["instruction"]
        if example.get("input"):
            if example["input"].strip():
                question += f"\n{example['input']}"
        answer = example["output"]
            
        q_ids = tokenizer(question, truncation=True, 
                max_length=max_input_length).input_ids
        a_ids = tokenizer(answer, truncation=True, 
                max_length=max_output_length).input_ids
        input_ids = q_ids + [tokenizer.eos_token_id] + a_ids + [tokenizer.eos_token_id]
        question_length = len(q_ids) + 1
        labels = [-100] * question_length + input_ids[question_length:]
        return dict(input_ids=input_ids, labels=labels)
    
    return data.map(tokenize_function, remove_columns=column_names)

def make_supervised_data_module(data_args, tokenizer):
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.eval_path is not None:
        train_dataset = build_dataset(data_args.data_path, tokenizer,
                        data_args.max_input_length, data_args.max_output_length)
        eval_dataset = build_dataset(data_args.eval_path, tokenizer,
                        data_args.max_input_length, data_args.max_output_length)
    elif data_args.test_size > 0:
        data = load_dataset(path="json", data_files=data_args.data_path)
        train_test = data["train"].train_test_split(test_size=data_args.test_size,
                        shuffle=True, seed=training_args.seed)
        train_dataset = build_dataset(train_test["train"], tokenizer,
                        data_args.max_input_length, data_args.max_output_length)
        eval_dataset = build_dataset(train_test["test"], tokenizer,
                        data_args.max_input_length, data_args.max_output_length)
    else:
        train_dataset = build_dataset(data_args.data_path, tokenizer,
                        data_args.max_input_length, data_args.max_output_length)
        eval_dataset = None
    
    data_collator = DataCollator(pad_token_id=tokenizer.pad_token_id)
    return dict(train_dataset=train_dataset, 
                eval_dataset=eval_dataset, 
                data_collator=data_collator)
        
def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set seed
    set_seed(training_args.seed)
    
    # Load model and tokenizer
    model = BloomForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )
    tokenizer = BloomTokenizerFast.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right")

    tokenizer.pad_token_id = 0
    
    data_module = make_supervised_data_module(data_args=data_args, tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    
if __name__ == "__main__":
    train()

