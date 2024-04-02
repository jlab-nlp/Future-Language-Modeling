import torch
from math import log,sqrt
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.data.data_collator import DataCollatorMixin, _torch_collate_batch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding
from transformers import AutoTokenizer
from collections import defaultdict
tokenizer_kwargs = {'cache_dir': "transformer_cache", 'use_fast': True, 'revision': 'main', 'use_auth_token': None}
tokenizer = AutoTokenizer.from_pretrained("gpt2", **tokenizer_kwargs)


def get_unigram_freq_prob_tensor(unigram_file="ngrambyyear/unigram_frequency-2019-cut-2019-tokenizer-tokenize.txt",
                                 use_log=True):
    unigram_freq_dict = {}
    total = 0
    with open(unigram_file) as f:
        lines = f.readlines()
        for line in lines:
            token_freq = line.strip("\n").split("\t")
            token = token_freq[1]
            freq = int(token_freq[2])
            total += freq
            unigram_freq_dict[token] = freq
    unigram_freq_prob_dict = {}
    vocab = tokenizer.vocab
    for token in vocab.keys():
        if token in unigram_freq_dict.keys():
            unigram_freq_prob_dict[vocab[token]] = log(
                (unigram_freq_dict[token] + 1) * 1.0 / (total + len(tokenizer)))
        else:
            unigram_freq_prob_dict[vocab[token]] = log(1 * 1.0 / (total + len(tokenizer)))
    bias = [v for k, v in sorted(unigram_freq_prob_dict.items(), key=lambda x: x[0])]
    bias_tensor = torch.tensor(bias, dtype=torch.float32)
    return bias_tensor


def get_unigram_freq_prob_year_tensor(start_year=2000, end_year=2019, cut_year=2000, is_scalling=True):
    list_of_year_tensor = []
    for year in range(start_year, end_year+1, 1):
        unigram_file = f"ngrambyyear/unigram_frequency-{year}-cut-{cut_year}-tokenizer-tokenize.txt"
        bias_tensor = get_unigram_freq_prob_tensor(unigram_file)
        # scaling
        if is_scalling:
            bound = 1 / sqrt(768)  # divide by the embedding size
            new_bias = (bound + bound) * (bias_tensor - torch.min(bias_tensor)) / (torch.max(bias_tensor) - torch.min(bias_tensor)) - bound * torch.ones(bias_tensor.size())
            # [a, b] (b-a)*(values - min(values))/(max(values) - min(values)) + a
            list_of_year_tensor.append(new_bias)
        else:
            list_of_year_tensor.append(bias_tensor)
    return torch.stack(list_of_year_tensor)



@dataclass
class DataCollatorForLanguageModelingWithNgram(DataCollatorMixin):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

    </Tip>"""

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }
        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

if __name__ == '__main__':
    get_unigram_freq_prob_tensor()