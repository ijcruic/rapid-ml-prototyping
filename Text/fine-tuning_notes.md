# Using Accelerate for Fine-tuning Large Language Models  
  
Accelerate is a utility from Hugging Face that simplifies distributed and mixed-precision training for `transformers` models.  
  
## Specifying GPUs  
  
You can specify the GPUs to use by setting the `CUDA_VISIBLE_DEVICES` environment variable when running your script. For example, if you want to use GPUs 3, 4, and 5, you would run your script like this:  

```
CUDA_VISIBLE_DEVICES=3,4,5 python llm_finetuning_for_stance.py
```

At current time, this method works better for splitting a model across multiple GPUs, when combined with `device_map=...`. Alternatively, you can use the `accelerate` command to launch your script on specific GPUs. Here's how to do it:  

```
accelerate launch --gpu_ids="3,4,5" --num_processes=1 --mixed_precision="fp16" llm_finetuning_for_stance.py
```

__Note:__ when using accelerate, you also need to drop the `device_map=...` argument from the AutoModel class.

## Loading Adapters  
  
Here's a code snippet that shows how to load adapters for use:  
  
```python  
# Load the configuration, model, and tokenizer    
config = PeftConfig.from_pretrained(MODEL_PATH)    
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, trust_remote_code=True, device_map=gpu_index)    
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)    
  
# Load the pretrained model    
model = PeftModel.from_pretrained(model, MODEL_PATH).merge_and_unload(progressbar=True)    
  
# Define a pipeline for text generation    
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,     
                eos_token_id=tokenizer.eos_token_id,    
                pad_token_id=tokenizer.eos_token_id,    
                max_new_tokens=100)
```

In this code, the PeftConfig and PeftModel are used to load the adapters (also known as the PEFT configuration) from the specified MODEL_PATH. Then, the model is used to create a text generation pipeline.  