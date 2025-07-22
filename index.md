## Basic 

[gradient_descent_and_derivative.ipynb](https://colab.research.google.com/drive/18IfySN0wKFizTiFYf9g0TGxjBFytah_z){: .btn .fs-3 }

## LLM

###  Tokenizer

[llm_tokenizer.ipynb](https://colab.research.google.com/drive/1YXoxLfQ5CXiB0GivAuoe0RR1TVh-Yabe){: .btn .fs-3 }

  text input -> tensor -> LLM -> tensor -> text ouput 

### Fine tuning with PEFT (Parameter-Efficient Fine-Tuning)

[llm_finetuning_lora.ipynb](https://colab.research.google.com/drive/1Eb8Ry7W3P2XBwhYWltg50z_aLaja2vYb){: .btn .fs-3 }

  LoRA（Low-Rank Adaptation）Fine tuning \
  Partial Fine-Tuning (Adapter-based or Layer Freezing), applies LoRA to specific layers  

### 3 types fine tuning: Prompt, Prefix, Lora

[llm_fine_tuning_prompt_prefix_lora.ipynb](https://colab.research.google.com/drive/17UxHuZR7-4CKXqidlhpJEAN6bVG2awGp#scrollTo=OwoxB86g1Frp){: .btn  .fs-3 }

  * Prompt fine tuning
  
  key configuration:
  
  ```python
  # prompt_tuning_init=PromptTuningInit.RANDOM,   # The added virtual tokens are initializad with random numbers or text
  prompt_tuning_init=PromptTuningInit.TEXT,
  prompt_tuning_init_text='a',
  num_virtual_tokens=6,                           # Number of virtual tokens to be prepend and trained.
  ```
  
  * Prefix fine tuning

  ```python
  num_virtual_tokens=30,   # Longer prefixes can increase capacity but risk overfitting with limited data
  prefix_projection=True,  # Adds a two-layer MLP projection over the prefix embeddings
  ```

  * Lora fine tuning

  ```python
  r=8,                           # higher rank (e.g. r=16) gives more capacity but uses more memory. 8 is good default value
  lora_alpha=32,                 # how strongly the adapters modify the frozen weights. Lowering alpha (e.g. lora_alpha=16) makes updates softer
  lora_dropout=0.1,              # 0.1 means Randomly drops 10% of the LoRA activations during training to prevent overfitting
  target_modules=["c_attn"],     # by default, LoRA targets the attention projection layers (e.g., q_proj, v_proj). can target just that for minimal intervention if we know the exact layer name (like c_attn in GPT-2),
  ```



{: .note }
This is a note-style paragraph. It will appear in a styled callout box.





