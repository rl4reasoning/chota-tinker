from datetime import datetime

from datasets import load_dataset
import json, zlib, base64, pickle

# SDPO paper (arXiv:2601.20802) uses LCB v6: Feb 1 - Apr 30, 2025 (131 problems).
# Matches SDPO: revision=refs/pr/6, contest_date >= Feb 1 and < May 1.
# See https://github.com/lasgroup/SDPO/blob/main/data/utils/livecodebench.py
SDPO_LCB_TEST_CUTOFF = datetime(2025, 2, 1)
SDPO_LCB_UNTIL = datetime(2025, 5, 1)
SOURCE_TAG = "lcbv6_feb_may_2025"
HUB_DATASET_NAME = "lcb_v6_feb_may_2025_formatted"


def _parse_contest_date(s) -> datetime | None:
    """Parse contest_date to datetime for comparison."""
    if s is None:
        return None
    if isinstance(s, datetime):
        return s
    s = str(s).strip().replace("T", " ")[:19]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[: len(fmt)], fmt)
        except (ValueError, TypeError):
            continue
    return None


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
    info_dict = {"tests": json.dumps(tests_dict), "source": SOURCE_TAG}

    return {
        "info": json.dumps(info_dict),
        "question": question,
        "avg@8_qwen3_4b_instruct_2507": 0.0,
    }

def _in_sdpo_range(example) -> bool:
    """SDPO filter: contest_date >= Feb 1, 2025 and < May 1, 2025."""
    dt = _parse_contest_date(example.get("contest_date"))
    if dt is None:
        return False
    return SDPO_LCB_TEST_CUTOFF <= dt < SDPO_LCB_UNTIL


def main():
    # SDPO uses revision=refs/pr/6 (see HF discussions/5) + date filter -> 131 problems
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        revision="refs/pr/6",
    )
    ds = ds.filter(_in_sdpo_range, desc="Filtering SDPO LCB range (Feb 1 - Apr 30, 2025)")
    print(f"Filtered to {len(ds)} problems (SDPO LCB: Feb 1 - Apr 30, 2025)")

    new_ds = ds.map(
        process_lcb_example,
        remove_columns=ds.column_names,   # drop original big columns
        desc=f"Formatting {SOURCE_TAG}",
        keep_in_memory=False,             # IMPORTANT: write to disk, not RAM
        writer_batch_size=50,             # smaller batches => lower peak RAM
    )

    # Reduce peak memory during push by forcing smaller shards
    new_ds.push_to_hub(
        f"anirudhb11/{HUB_DATASET_NAME}",
        private=False,
        max_shard_size="200MB",
    )

if __name__ == "__main__":
    main()
