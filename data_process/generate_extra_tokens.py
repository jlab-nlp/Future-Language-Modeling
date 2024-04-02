from nltk import word_tokenize
from data_process import read_csv
from transformers import AutoTokenizer
from tqdm import tqdm
from langdetect import detect, lang_detect_exception
from collections import Counter
from transformers import BasicTokenizer
import pickle

def containsNumber(value):
    return any([char.isdigit() for char in value])

if __name__ == '__main__':
    seen_tokens = Counter()
    papers = read_csv(input_file="papers-filtered-final-without-semeval.csv")
    tokenizer = BasicTokenizer(do_lower_case=False)
    for paper in tqdm(papers, total=len(papers)):
        if "Ó" not in paper.abstract:
            tokens = tokenizer.tokenize(paper.abstract)
            if "tran" in tokens:
                print(paper.abstract)
                exit(0)
            seen_tokens.update(tokens)

    incorrect_tokens = []
    for token in tqdm(seen_tokens.keys(), total=len(seen_tokens.keys())):
        try:
            if containsNumber(token) or detect(token) == 'ko':
                if token not in incorrect_tokens:
                    incorrect_tokens.append(token)
        except lang_detect_exception.LangDetectException:
            print(f"No features in text. token: {token}")
            if token not in incorrect_tokens:
                incorrect_tokens.append(token)
            continue
    with open("incorrect_tokens_basic.pickle", "wb") as f:
        pickle.dump(incorrect_tokens, f)
    with open("seen_tokens_basic.pickle", "wb") as f:
        pickle.dump(seen_tokens, f)
    exit(0)

    no_bpe_vocab = set()
    with open("incorrect_tokens_basic.pickle", "rb") as f:
        incorrect_tokens = pickle.load(f)
    with open("seen_tokens_basic.pickle", "rb") as f:
        seen_tokens = pickle.load(f)
    for token in tqdm(seen_tokens.keys(), total=len(seen_tokens)):
        if token not in incorrect_tokens and seen_tokens[token] > 10:
            no_bpe_vocab.add(token)
    print(len(no_bpe_vocab))
        # if not containsNumber(token) and seen_tokens[token] > 5:
        #     try:
        #         if detect(token) != 'ko':
        #             no_bpe_vocab.add(token)
        #
    tokenizer_kwargs = {'cache_dir': None, 'use_fast': True, 'revision': 'main', 'use_auth_token': None}
    tokenizer = AutoTokenizer.from_pretrained("gpt2", **tokenizer_kwargs)
    bpe_vocab = set(list(tokenizer.vocab.keys()))
    extra_tokens = list(no_bpe_vocab.difference(bpe_vocab))
    special_tokens_dict = {'additional_special_tokens': extra_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(num_added_toks)
    # special_tokens_dict = {'additional_special_tokens': extra_tokens[25000:]}
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # tokenizer.save_pretrained("saved_extended_tokenizer")
    print(len(tokenizer))
    print(tokenizer.tokenize("Transformer is a good language modeling architecture"))
    print(tokenizer.vocab)





