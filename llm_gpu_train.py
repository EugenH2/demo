!pip install bitsandbytes
!pip install peft
!pip install trl
!pip install transformers
!pip install -q -U datasets==2.17.0

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd

from transformers import TrainingArgumentsmodel_id = "/kaggle/input/gemma/transformers/7b-it/2"
tokenizer_id = "/kaggle/input/gemmatokenizer/gemmatokenizer"

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    #bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.padding_side = 'right' # to prevent warnings


TEST_DF_FILE = '/kaggle/input/gemma-rewrite-nbroad/nbroad-v1.csv'

tdf = pd.read_csv(TEST_DF_FILE, nrows=None, usecols=['id', 'original_text', 'rewritten_text'])
sub = pd.read_csv(SUB_DF_FILE, nrows=None, usecols=['id', 'rewrite_prompt'])

result = pd.concat([tdf,sub['rewrite_prompt']],axis=1)#join='inner'

# Convert dataset to OAI messages
system_message = """Given are 2 essays, the Rewritten essay was created from the Original essay using the google Gemma model. 
You are trying to understand how the original essay was transformed into a new version. 
Analyzing the changes in style, theme, etc., please come up with a prompt that must have been used to guide the transformation from the original to the rewritten essay. 
Start directly with the prompt, that's all I need. Output should be one line ONLY."""

def create_conversation(sample):
    userpromt = f"""
Original Essay:
\"""{sample["original_text"]}\"""

Rewritten Essay:
\"""{sample["rewritten_text"]}\"""
"""
    
    return {
        "messages": [
            {"role": "system", "content": system_message },
            {"role": "user", "content": userpromt},
            {"role": "assistant", "content": sample["rewrite_prompt"]}
        ]
    }



dataset = Dataset.from_pandas(result)
# Convert dataset to OAI messages
dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)


peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=6,
        bias="none",
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
)

max_seq_length = 1512 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    args=TrainingArguments(
        output_dir="gemma7bIt", # directory to save and repository id
        num_train_epochs=3,
        per_device_train_batch_size=1,          # batch size per device during training
        gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        gradient_checkpointing_kwargs={'use_reentrant':False},
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=100,                      # log every 10 steps
        save_strategy="epoch",             	# save checkpoint every epoch
        fp16=True,
        learning_rate=2e-4,                     # learning rate, based on QLoRA paper
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=False,                      # push model to hub
        report_to="none",                # report metrics to tensorboard
    ),
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    dataset_kwargs={
        "add_special_tokens": False, # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }
)

trainer.train()
# save model
trainer.save_model()

