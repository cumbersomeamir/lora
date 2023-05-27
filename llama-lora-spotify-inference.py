import os
import bitsandbytes as bnb
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
peft_model_id = "Amirkid/llama-lora-spotify"

# Load the base model configuration
config = PeftConfig.from_pretrained(peft_model_id)

# Load the base model with the desired parameters
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')

# Load the tokenizer for the base model
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

# Prepare the input
batch = tokenizer("Amir opened up with this joke", return_tensors='pt')

with torch.cuda.amp.autocast():
    output_tokens = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_new_tokens=50)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
