"""
Sample one problem from the prompt.txt file using chota-tinker.

Start vLLM server first:
    vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8000 --gpu-memory-utilization 0.6

Shut down vLLM server:
    Ctrl+C in the terminal running the server

Usage:
    python sample_api.py
"""

from chota_tinker import ServerSamplingClient, SamplingParams, ModelInput
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "Qwen/Qwen3-1.7B"
prompt_file = "prompt.txt" # load prompt from file, edit this file to change the prompt

# Load the prompt from prompt.txt
with open(prompt_file, "r") as f:
    prompt = f.read()

# Connect to vLLM server
sampling_client = ServerSamplingClient("http://localhost:8000")

# Get the tokenizer from transformers
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a prompt - use chat template if available, otherwise raw prompt
if tokenizer.chat_template:
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
else:
    # Base model without chat template - use raw prompt
    prompt_text = prompt

input_ids = tokenizer(prompt_text)["input_ids"]

# Sample from the model
result = sampling_client.sample(
    ModelInput.from_ints(input_ids),
    SamplingParams(max_tokens=500, temperature=0.6),
    num_samples=1,
)

response = result.sequences[0].text

# Print prompt and response
print(prompt_text, end="")
print(response)
