import os

from datasets import load_dataset
import json, zlib, base64, pickle

def decode_private_tests(private_raw: str):
    # First try direct JSON
    try:
        return json.loads(private_raw)
    except json.JSONDecodeError:
        pass

    # Otherwise assume base64(zlib(pickle(JSON_STRING_OR_OBJ)))
    data = base64.b64decode(private_raw.encode("utf-8"))
    obj = pickle.loads(zlib.decompress(data))

    # The pickled object might already be a list/dict or might be a JSON string/bytes.
    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode("utf-8", errors="replace")
    if isinstance(obj, str):
        return json.loads(obj)
    return obj  # already parsed python object

def process_lcb_example(example):
    question = example["question_content"] or ""

    starter = example.get("starter_code")
    
    if starter and len(starter) > 0:
        question += f"\n\nYou will use the following starter code to write the solution to the problem and enclose your code within ```python delimiters.\n\n```python\n{starter}\n```"
    else:
         question += "\n\nRead the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within ```python delimiters."

    public_test_cases = json.loads(example["public_test_cases"])
    private_test_cases = decode_private_tests(example["private_test_cases"])

    all_test_cases = public_test_cases + private_test_cases
    inputs = [tc["input"] for tc in all_test_cases]
    outputs = [tc["output"] for tc in all_test_cases]

    metadata = json.loads(example["metadata"])
    fn_name = metadata.get("func_name", None)

    tests_dict = {"inputs": inputs, "outputs": outputs, "fn_name": fn_name}
    info_dict = {"tests": json.dumps(tests_dict), "source": "lcbv6"}

    return {
        "info": json.dumps(info_dict),
        "question": question,
        "avg@8_qwen3_4b_instruct_2507": 0.0,
    }

def main():
    ds = load_dataset("livecodebench/code_generation_lite", version_tag="release_v6", split="test")

    new_ds = ds.map(
        process_lcb_example,
        remove_columns=ds.column_names,   # drop original big columns
        desc="Formatting LCB v6",
        keep_in_memory=False,             # IMPORTANT: write to disk, not RAM
        writer_batch_size=50,             # smaller batches => lower peak RAM
    )

    # Reduce peak memory during push by forcing smaller shards
    new_ds.push_to_hub(
        "anirudhb11/lcb_v6_formatted",
        private=False,
        max_shard_size="200MB",
    )

if __name__ == "__main__":
    main()
