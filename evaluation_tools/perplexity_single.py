#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0G
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""

import argparse
import logging
import math
import numpy as np
import torch
from tqdm import tqdm
import pickle


from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

from model import (
    GPT2LMHeadUNIGRAMModel,
    GPT2LMHeadUNIGRAMRNNModel,
    GPT2LMHeadUNIGRAMRNNWindowModel,
    GPT2LMHeadUNIGRAMRNNWindowWeightModel,
    GPT2LMHeadVocabReprRNNWindowWeightModel,
    GPT2LMHeadUNIGRAMVocabReprRNNWindowWeightModel,
    GPT2LMHeadVocabReprRNNWindowSigmoidAttentionModel,
    GPT2LMHeadVocabReprRNNWindowSigmoidAttentionDotModel
)
from ngramdatacolletor import get_unigram_freq_prob_tensor, get_unigram_freq_prob_year_tensor
from torch.nn import Parameter
from data_process import normalized, stopwords

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2-unigram": (GPT2LMHeadUNIGRAMModel, GPT2Tokenizer),
    "gpt2-unigram-rnn": (GPT2LMHeadUNIGRAMRNNModel, GPT2Tokenizer),
    "gpt2-unigram-rnn-window": (GPT2LMHeadUNIGRAMRNNWindowModel, GPT2Tokenizer),
    "gpt2-unigram-rnn-window-weight": (GPT2LMHeadUNIGRAMRNNWindowWeightModel, GPT2Tokenizer),
    "gpt2-vocab-repr-rnn-window-weight": (GPT2LMHeadVocabReprRNNWindowWeightModel, GPT2Tokenizer),
    "gpt2-unigram-vocab-repr-rnn-window-weight": (GPT2LMHeadUNIGRAMVocabReprRNNWindowWeightModel, GPT2Tokenizer),
    "gpt2-vocab-repr-rnn-window-sigmoid-attention": (GPT2LMHeadVocabReprRNNWindowSigmoidAttentionModel, GPT2Tokenizer),
    "gpt2-vocab-repr-rnn-window-sigmoid-attention-dot": (GPT2LMHeadVocabReprRNNWindowSigmoidAttentionDotModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


#def get_nonstopwords_indexes(tokens, device):
#    return torch.tensor([id for id, token in enumerate(tokens) if token not in stopwords], dtype=torch.long, device=device)

def get_nonstopwords_indexes(tokens, device):
    nonstopwords_indexes = []
    for id, token in enumerate(tokens):
        if "Ġ" in token:
            if token[1:].lower() not in stopwords and len(token[1:]) > 4:
                nonstopwords_indexes.append(id)
        else:
            if token.lower() not in stopwords and len(token) > 4:
                nonstopwords_indexes.append(id)
    return torch.tensor(nonstopwords_indexes, dtype=torch.long, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default="  ", help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--output_dir", type=str, default="", help="Ourput_dir")
    parser.add_argument("--kpp", action="store_true", help="Compute key phrases perplexity")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--beta", type=float, default=2.0, help="beta to enlarge ngram difference")
    parser.add_argument("--window_size", type=int, default=3, help="beta to enlarge ngram difference")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--use_log", action="store_true", help="use log probability to compute the bias.")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Whether to scale the ngram probability",
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    if args.model_type == "gpt2-unigram":
        #model = model_class.from_pretrained(args.model_name_or_path, ngram_bias=get_unigram_freq_prob_tensor(
        #    unigram_file="ngrambyyear/unigram_frequency-2020-cut-2019-tokenizer-tokenize.txt", use_log=args.use_log))
        #bias = get_unigram_freq_prob_tensor(
        #    unigram_file="ngrambyyear/unigram_frequency-2020-cut-2019-tokenizer-tokenize.txt", use_log=args.use_log)
        #train_bias = get_unigram_freq_prob_tensor(
        #    unigram_file="ngrambyyear/unigram_frequency-2019-cut-2019-tokenizer-tokenize.txt", use_log=args.use_log)
        #bias = bias + args.beta * (bias - train_bias)
        #fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(model.lm_head.weight)
        #bound = 1 / math.sqrt(fan_in)
        # scaling
        #new_bias = (bound + bound) * \
        #           (bias - torch.min(bias)) / (torch.max(bias) - torch.min(bias)) - bound * torch.ones(bias.size())
        #model.lm_head.bias = Parameter(1.0 * new_bias)
        #ngrambyyear/unigram_frequency-2019-cut-2019-start-2017-tokenizer-tokenize.txt
        model = model_class.from_pretrained(args.model_name_or_path, ngram_bias=get_unigram_freq_prob_tensor(
            unigram_file="ngrambyyear/unigram_frequency-2019-cut-2019-start-2017-tokenizer-tokenize.txt", use_log=True))
    elif args.model_type == "gpt2-unigram-rnn":
        model = model_class.from_pretrained(args.model_name_or_path,
                                            unigram_year_tensor=
                                            get_unigram_freq_prob_year_tensor(start_year=2000,
                                                                              end_year=2019,
                                                                              cut_year=2000,
                                                                              is_scalling=True))
    elif args.model_type == "gpt2-unigram-rnn-window":
        model = model_class.from_pretrained(args.model_name_or_path,
                                            unigram_year_tensor=
                                            get_unigram_freq_prob_year_tensor(start_year=2000,
                                                                              end_year=2020,
                                                                              cut_year=2000,
                                                                              is_scalling=False),
                                            window_size=args.window_size)
    elif args.model_type == "gpt2-unigram-rnn-window-weight":
        model = model_class.from_pretrained(args.model_name_or_path,
                                            unigram_year_tensor=
                                            get_unigram_freq_prob_year_tensor(start_year=2000,
                                                                              end_year=2019,
                                                                              cut_year=2000,
                                                                              is_scalling=False),
                                            window_size=args.window_size)
    elif args.model_type == "gpt2-vocab-repr-rnn-window-weight":
        model = GPT2LMHeadVocabReprRNNWindowWeightModel.from_pretrained(args.model_name_or_path,
                                                                        year_vocab_repr=torch.load(
                                                                            "year_vocab_repr.pt"),
                                                                        window_size=args.window_size)
    elif args.model_type == "gpt2-unigram-vocab-repr-rnn-window-weight":
        model = GPT2LMHeadUNIGRAMVocabReprRNNWindowWeightModel.from_pretrained(args.model_name_or_path,
                                                                               unigram_year_tensor=
                                                                               get_unigram_freq_prob_year_tensor(
                                                                                   start_year=2000,
                                                                                   end_year=2019,
                                                                                   cut_year=2000,
                                                                                   is_scalling=False),
                                                                               year_vocab_repr=torch.load(
                                                                                   "year_vocab_repr.pt"),
                                                                               window_size=args.window_size)
    elif args.model_type == "gpt2-unigram-vocab-repr-by-doc-rnn-window-weight":
        model = GPT2LMHeadUNIGRAMVocabReprRNNWindowWeightModel.from_pretrained(args.model_name_or_path,
                                                                               unigram_year_tensor=
                                                                               get_unigram_freq_prob_year_tensor(
                                                                                   start_year=2000,
                                                                                   end_year=2019,
                                                                                   cut_year=2000,
                                                                                   is_scalling=False),
                                                                               year_vocab_repr=torch.load(
                                                                                   "year_vocab_by_doc_repr.pt"),
                                                                               window_size=args.window_size)
    elif args.model_type == "gpt2-vocab-repr-rnn-window-sigmoid-attention":
        model = GPT2LMHeadVocabReprRNNWindowSigmoidAttentionModel.from_pretrained(args.model_name_or_path,
                                                                                  unigram_year_tensor=
                                                                                  get_unigram_freq_prob_year_tensor(
                                                                                      start_year=2000,
                                                                                      end_year=2019,
                                                                                      cut_year=2000,
                                                                                      is_scalling=True),
                                                                                  year_vocab_repr=torch.load(
                                                                                      "year_vocab_repr.pt"),
                                                                                  window_size=args.window_size,
                                                                                  alpha=0.0001)
    elif args.model_type == "gpt2-vocab-repr-rnn-window-sigmoid-attention-dot":
        model = GPT2LMHeadVocabReprRNNWindowSigmoidAttentionDotModel.from_pretrained(args.model_name_or_path,
                                                                                     unigram_year_tensor=
                                                                                     get_unigram_freq_prob_year_tensor(
                                                                                         start_year=2000,
                                                                                         end_year=2019,
                                                                                         cut_year=2000,
                                                                                         is_scalling=True),
                                                                                     year_vocab_repr=torch.load(
                                                                                         "year_vocab_repr.pt"),
                                                                                     window_size=args.window_size,
                                                                                     do_norm=False)
    else:
        model = model_class.from_pretrained(args.model_name_or_path)

    model.to(args.device)
    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)
    max_length= model.config.n_positions
    encodings = []
    with open("papers_2020.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            encoding = tokenizer(line.strip(), truncation=True, max_length=max_length, return_tensors="pt")
            encodings.append(encoding)
    model_kwargs = {"previous_year_index": torch.tensor([19], dtype=torch.long),
                    "is_generate": True}
    ppl_scores = []
    for i, encode_repr in tqdm(enumerate(encodings), total=len(encodings)):
        input_ids = encode_repr.input_ids.to(args.device)
        target_ids = input_ids.clone()
        trg_len = min(max_length, encode_repr.input_ids.size(1))
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            if args.kpp:
                if args.model_type == "gpt2" or args.model_type == "gpt2-unigram":
                    outputs = model(input_ids, labels=target_ids)  # , **model_kwargs)
                else:
                    outputs = model(input_ids, labels=target_ids, **model_kwargs)
                lm_logits = outputs[1]

                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                # Flatten the tokens
                non_stop_labels_indexes = get_nonstopwords_indexes(
                    tokenizer.convert_ids_to_tokens(shift_labels.squeeze().tolist()), device=args.device)
                shift_labels = shift_labels[:, non_stop_labels_indexes]
                shift_logits = shift_logits[:, non_stop_labels_indexes, :]

                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                ppl_scores.append(loss.item())
                return ppl_scores
            else:
                if args.model_type == "gpt2" or args.model_type == "gpt2-unigram":
                    outputs = model(input_ids, labels=target_ids)
                else:
                    outputs = model(input_ids, labels=target_ids, **model_kwargs)
                lm_logits = outputs[1]
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                ppl_scores.append(loss.item())
    return args.output_dir, args.model_name_or_path.split("/")[1], ppl_scores



if __name__ == "__main__":
    output_dir, model_name, ppl_scores = main()
    with open(output_dir+"/"+model_name+".pickle", "wb") as f:
        pickle.dump(ppl_scores, f)
