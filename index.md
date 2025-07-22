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

  * Lora fine tuning


> [!NOTE]  
> test

{: .note }
test



