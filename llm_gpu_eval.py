import datetime
start_time = datetime.datetime.now() 
import peft
from datasets import Dataset
import pandas as pd
import torch
from peft import AutoPeftModelForCausalLM
from transformers import  AutoTokenizer, pipeline#, AutoModelForCausalLM


TEST_DF_FILE = '/kaggle/input/llm-prompt-recovery/test.csv'
tdf = pd.read_csv(TEST_DF_FILE, nrows=None, usecols=['id', 'original_text', 'rewritten_text'])

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
            {"role": "user", "content": userpromt}
        ]
    }


dataset = Dataset.from_pandas(tdf)
dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)


peft_model_id = "/kaggle/input/gemma7bit/gemma7bIt/checkpoint-1623"

# Load Model with PEFT adapter
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", torch_dtype=torch.bfloat16)
model.eval()


pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)#, batch_size=4)
# get token id for end of conversation
eos_token = tokenizer("<|im_end|>",add_special_tokens=False)["input_ids"][0]

def test_inference(dataset):
    prompt = pipe.tokenizer.apply_chat_template(dataset["messages"], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=128, do_sample=True, temperature=0.01, top_k=50, top_p=0.95, eos_token_id=eos_token)

    return outputs[0]['generated_text'][len(prompt):].strip()


res = []
for i, s in enumerate(dataset):
    try:
        res.append([tdf["id"].iloc[i], test_inference(s)])
        
    except Exception as e:
        #res.append([tdf["id"].iloc[i], DEFAULT_TEXT])


sub = pd.DataFrame(res, columns=['id', 'rewrite_prompt'])
sub.to_csv("submission.csv", index=False)
