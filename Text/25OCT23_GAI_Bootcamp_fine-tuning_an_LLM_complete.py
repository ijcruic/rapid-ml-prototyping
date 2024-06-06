# Import necessary libraries and modules  
import pandas as pd, os  
import accelerate, torch  
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig  
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM    
from peft import LoraConfig    
from datasets import Dataset    

# Load the training data from a CSV file into a pandas DataFrame  
df = pd.read_csv("/home/jovyan/wire/WIREUsers/icruickshank/LLM-Stance-Labeling/all_data_training.csv")

# we are going to downsample just for the code walkthrough
df = df.sample(n=1000, random_state=1)  

# Convert the pandas DataFrame to a Hugging Face Dataset  
dataset = Dataset.from_pandas(df)  
  
# Load the pre-trained model and the associated tokenizer   
# device_map='auto' will spread the model across multiple GPUs if available  
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# If a quantization_config is provided, the model will be quantized. There re several options for
# this type of quantization
'''
quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
        )
        
quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
        ) 
'''

quantization_config= None

tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,   
    quantization_config=quantization_config)  



  
# Ensure the tokenizer has a pad token, and set padding to the right  
if not tokenizer.pad_token:    
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})  
tokenizer.padding_side = 'right'  

# Define a function to format the training data. For this, we need to make sure that the prompts get formatted according to the models .generate function
# some models place additional tags when they generate text. If we don't do this, the trainer will not succesfully find the text that the model
# is supposed to produce.
def formatting_prompts_func(example):  
    output_texts = []  
    for i in range(len(example['context'])):  
        text = f"<s> {example['context'][i]}\n entity: {example['event'][i]}\n statement: {example['full_text'][i]}\n </s> <s> stance: {example['train_stance'][i]}"  
        output_texts.append(text)  
    return output_texts    

# Define the response template for data collator  
response_template = "stance:"    
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)    




  
# Set up the training arguments for the trainer  
training_args = TrainingArguments(    
    output_dir='training_results',
    num_train_epochs=1,  
    auto_find_batch_size=True, #handy function for finding the batch size instead of specifying
    warmup_steps=50,
    learning_rate= 1e-5,  
    gradient_accumulation_steps=12,
    weight_decay=0.01,
    save_steps = 50,
    logging_dir='training_results/logs',
    fp16=True #mixed precision training
)  


  
# Set up the configuration for PEFT  
peft_config = LoraConfig(    
    r=64,    
    lora_alpha=16,    
    lora_dropout=0.05,    
    bias="none",  
    task_type="CAUSAL_LM",  
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  
)    
  
# Initialize the SFTTrainer  
trainer = SFTTrainer(  
    model,  
    tokenizer=tokenizer,  
    args=training_args,  
    train_dataset=dataset,   
    formatting_func=formatting_prompts_func,    
    data_collator=collator,    
    peft_config=peft_config,  
    max_seq_length = 500  
)    

trainer.train()

trainer.save_model("training_results/stance_tuned_model")