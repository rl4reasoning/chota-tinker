import tinker
from tinker import types # Convert examples into the format expected by the training client
from transformers import AutoTokenizer

service_client = tinker.ServiceClient()

# model_name = "meta-llama/Llama-3.2-3B"
model_name = "Qwen/Qwen3-4B-Instruct-2507" # this does not have thinking :(
# model_name = "openai/gpt-oss-20b"

# Create a sampling client directly from the base model
sampling_client = service_client.create_sampling_client(base_model=model_name)

# Get the tokenizer from transformers
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the prompt from prompt.txt
with open("prompt.txt", "r") as f:
    _prompt = f.read()

# Create a prompt using the chat template
messages = [
    {"role": "user", "content": _prompt}
]
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
input_ids = tokenizer(prompt_text)["input_ids"]

input_ids = types.ModelInput.from_ints(input_ids)
params = types.SamplingParams(max_tokens=500, temperature=0.6)

# Sample from the model
future = sampling_client.sample(input_ids, sampling_params=params, num_samples=1)
result = future.result()

response = tokenizer.decode(result.sequences[0].tokens)

# print prompt and response
print(prompt_text)
print(response)