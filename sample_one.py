"""
Sample one problem from the prompt.txt file using chota-tinker.

Start vLLM server first:
    vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000
"""

from chota_tinker import ServerSamplingClient, SamplingParams, ModelInput
from transformers import AutoTokenizer

# model_name = "meta-llama/Llama-3.2-3B"
model_name = "Qwen/Qwen3-4B-Instruct-2507"
# model_name = "openai/gpt-oss-20b"

# Connect to vLLM server
sampling_client = ServerSamplingClient("http://localhost:8000")

# Get the tokenizer from transformers
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the prompt from prompt.txt
with open("prompt.txt", "r") as f:
    _prompt = f.read()

# Create a prompt using the chat template
messages = [
    {"role": "user", "content": _prompt}
]
prompt_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
input_ids = tokenizer(prompt_text)["input_ids"]

# Sample from the model
result = sampling_client.sample(
    ModelInput.from_ints(input_ids),
    SamplingParams(max_tokens=500, temperature=0.6),
    num_samples=1,
)

response = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)

# Print prompt and response
print("=" * 50)
print("PROMPT:")
print("=" * 50)
print(prompt_text)
print("\n" + "=" * 50)
print("RESPONSE:")
print("=" * 50)
print(response)
