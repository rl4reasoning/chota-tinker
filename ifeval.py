"""IFEval dataset utilities for instruction-following RL training."""

import re
from datasets import load_dataset
from transformers import PreTrainedTokenizer

from chota_tinker import SamplingResult


# ============ Instruction Verifiers ============

def check_no_comma(text: str) -> bool:
    return "," not in text


def check_number_words(text: str, relation: str, num_words: int) -> bool:
    word_count = len(text.split())
    if relation == "at least":
        return word_count >= num_words
    elif relation == "at most":
        return word_count <= num_words
    elif relation == "less than":
        return word_count < num_words
    return False


def check_number_sentences(text: str, relation: str, num_sentences: int) -> bool:
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    count = len(sentences)
    if relation == "at least":
        return count >= num_sentences
    elif relation == "at most":
        return count <= num_sentences
    elif relation == "less than":
        return count < num_sentences
    return False


def check_keyword(text: str, keyword: str, frequency: int) -> bool:
    return text.lower().count(keyword.lower()) >= frequency


def check_instruction(text: str, inst_id: str, kwargs: dict) -> float:
    """Check if a single instruction is satisfied. Returns 1.0, 0.5, or 0.0."""
    try:
        if inst_id == "punctuation:no_comma":
            return 1.0 if check_no_comma(text) else 0.0
        elif inst_id == "length_constraints:number_words":
            return 1.0 if check_number_words(
                text, kwargs.get("relation", "at least"), kwargs.get("num_words", 100)
            ) else 0.0
        elif inst_id == "length_constraints:number_sentences":
            return 1.0 if check_number_sentences(
                text, kwargs.get("relation", "at least"), kwargs.get("num_sentences", 3)
            ) else 0.0
        elif inst_id == "keywords:frequency":
            return 1.0 if check_keyword(
                text, kwargs.get("keyword", ""), kwargs.get("frequency", 1)
            ) else 0.0
        else:
            return 0.5  # Unknown instruction - partial credit
    except Exception:
        return 0.5


def compute_reward(text: str, instruction_ids: list, kwargs_list: list) -> float:
    """Compute reward as fraction of instructions satisfied."""
    if not instruction_ids:
        return 1.0
    
    total_score = sum(
        check_instruction(text, inst_id, kwargs)
        for inst_id, kwargs in zip(instruction_ids, kwargs_list)
    )
    return total_score / len(instruction_ids)


# ============ Dataset Wrapper ============

class IFEvalDataset:
    """Wrapper for IFEval dataset with reward computation."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, split: str = "train"):
        self.dataset = load_dataset("google/IFEval", split=split)
        self.tokenizer = tokenizer
        self._iter = None
    
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        self._iter = iter(self.dataset)
        return self
    
    def __next__(self):
        return next(self._iter)
    
    def get_batch(self, batch_size: int) -> list[dict]:
        """Get a batch of examples, cycling if needed."""
        if self._iter is None:
            self._iter = iter(self.dataset)
        
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(self._iter))
            except StopIteration:
                self._iter = iter(self.dataset)
                batch.append(next(self._iter))
        return batch
    
    def format_prompts(self, batch: list[dict]) -> list[str]:
        """Format prompts with chat template."""
        formatted = []
        for ex in batch:
            messages = [{"role": "user", "content": ex["prompt"]}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted.append(text)
        return formatted
    
    def compute_rewards(
        self,
        batch: list[dict],
        results: list[SamplingResult],
    ) -> list[list[float]]:
        """Compute rewards for all samples in batch."""
        all_rewards = []
        
        for example, result in zip(batch, results):
            prompt_rewards = []
            for seq in result.sequences:
                text = self.tokenizer.decode(seq.tokens, skip_special_tokens=True)
                reward = compute_reward(
                    text,
                    example["instruction_id_list"],
                    example["kwargs"],
                )
                prompt_rewards.append(reward)
            all_rewards.append(prompt_rewards)
        
        return all_rewards
