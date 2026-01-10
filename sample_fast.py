"""
This can sample faster since models is hot in memory.

Usage:
    python sample_fast.py
"""

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "Qwen/Qwen3-1.7B"
prompt_file = "prompt.txt" # load prompt from file, edit this file to change the prompt

print(f"Loading {model_name}...")
llm = LLM(model=model_name, gpu_memory_utilization=0.6)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Model loaded! Edit {prompt_file} and press Enter to generate.\n")

sampling_params = SamplingParams(max_tokens=500, temperature=0.6)

while True:
    user_input = input(f"Press Enter to generate from {prompt_file}: ").strip()

    # Read prompt from file
    with open(prompt_file, "r") as f:
        prompt = f.read().strip()

    # Apply chat template if available
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = prompt

    # Generate
    outputs = llm.generate([prompt_text], sampling_params)
    response = outputs[0].outputs[0].text

    print(prompt_text, end="")
    print(response)

    print()
    print("=" * 50) # sep line :)
    print()
