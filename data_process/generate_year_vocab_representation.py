import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, AutoTokenizer
from tqdm import tqdm
from nltk import sent_tokenize
import time
from copy import deepcopy


def generate_year_vocab_representation(year):
    tokenizer = RobertaTokenizer.from_pretrained(f"saved_model/papers_{year}")
    model = RobertaForMaskedLM.from_pretrained(f"saved_model/papers_{year}", output_hidden_states=True).cuda()
    all_tokenized_sents = []
    max_len = 512
    with open(f"papers_{year}.txt") as f:
        lines = f.readlines()
        for line in tqdm(lines, total=len(lines), desc=f"load_{year}"):
            abstract = line.strip()
            sents = sent_tokenize(abstract)
            for sent in sents:
                tokenized_sent = tokenizer.tokenize(sent)
                if len(tokenized_sent) < max_len:
                    all_tokenized_sents.append(tokenized_sent)
    vocab_sentences = {}
    count_token = {}
    for sent_tokens in tqdm(all_tokenized_sents, total=len(all_tokenized_sents)):
        sent_tokens_ids = torch.tensor(tokenizer.convert_tokens_to_ids(sent_tokens), dtype=torch.long).unsqueeze(0)
        output = model(input_ids=sent_tokens_ids.cuda())[1][-1].squeeze().detach().cpu()
        for ind, token in enumerate(sent_tokens):
            token_representation = output[ind]
            if token in vocab_sentences:
                vocab_sentences[token] += token_representation
                count_token[token] += 1
            else:
                vocab_sentences[token] = token_representation
                count_token[token] = 1
    vocab_representation = {}
    for token in vocab_sentences.keys():
        vocab_representation[token] = vocab_sentences[token]/count_token[token]
    torch.save(vocab_representation, f"vocab_representation_{year}.th")
    return f"Finished year {year}!"


def generate_year_vocab_representation_org(year):
    tokenizer = RobertaTokenizer.from_pretrained(f"roberta-large")
    model = RobertaForMaskedLM.from_pretrained(f"roberta-large", output_hidden_states=True).cuda()
    all_tokenized_sents = []
    max_len = 512
    with open(f"papers_{year}.txt") as f:
        lines = f.readlines()
        for line in tqdm(lines, total=len(lines), desc=f"load_{year}"):
            abstract = line.strip()
            sents = sent_tokenize(abstract)
            for sent in sents:
                tokenized_sent = tokenizer.tokenize(sent)
                if len(tokenized_sent) < max_len:
                    all_tokenized_sents.append(tokenized_sent)
    vocab_sentences = {}
    count_token = {}
    for sent_tokens in tqdm(all_tokenized_sents, total=len(all_tokenized_sents)):
        sent_tokens_ids = torch.tensor(tokenizer.convert_tokens_to_ids(sent_tokens), dtype=torch.long).unsqueeze(0)
        output = model(input_ids=sent_tokens_ids.cuda())[1][-1].squeeze().detach().cpu()
        for ind, token in enumerate(sent_tokens):
            token_representation = output[ind]
            if token in vocab_sentences:
                vocab_sentences[token] += token_representation
                count_token[token] += 1
            else:
                vocab_sentences[token] = token_representation
                count_token[token] = 1
    vocab_representation = {}
    for token in vocab_sentences.keys():
        vocab_representation[token] = vocab_sentences[token]/count_token[token]
    torch.save(vocab_representation, f"vocab_representation_org_{year}.th")
    return f"Finished year {year}!"

gpt2_tokenizer = AutoTokenizer.from_pretrained(f"gpt2")
roberta_model = RobertaForMaskedLM.from_pretrained(f"roberta-large", output_hidden_states=True)
roberta_tokenizer = RobertaTokenizer.from_pretrained(f"roberta-large")


def generate_vocab_representation_tensor(vocab_repr):
    vocab = dict(sorted(gpt2_tokenizer.vocab.items(), key=lambda item: item[1])).keys()
    tensor_list = []
    for token in tqdm(vocab, total=len(vocab)):
        if token in vocab_repr:
            token_embedding = vocab_repr[token]
            token_embedding.requires_grad = False
            tensor_list.append(token_embedding)
        else:
            token_id = roberta_tokenizer.get_vocab()[token]
            token_embedding = roberta_model.get_input_embeddings().weight[token_id].detach()
            # token_embedding = torch.zeros(1024, dtype=torch.float32)
            token_embedding.requires_grad = False
            tensor_list.append(token_embedding)
    #vocab_repr_tensor = torch.stack(tensor_list)
    return tensor_list


def get_year_vocab_repr_tensor(start_year=2000, end_year=2021, prefix="vocab_representation"):
    tensor_lists = []
    for i in tqdm(range(start_year, end_year+1), total=end_year-start_year+1):
        vocab_repr = torch.load(f"{prefix}_{i}.th")
        tensor_lists.append(generate_vocab_representation_tensor(vocab_repr))
    return tensor_lists


def get_year_vocab_repr_diff_tensor(tensor_lists):
    tensor_diff_lists = [generate_vocab_representation_tensor({})]
    for year_id, year_tensor in tqdm(enumerate(tensor_lists), total=len(tensor_lists)):
        if year_id + 1 == len(tensor_lists):
            break
        current_year_tensor = year_tensor
        next_year_tensor = tensor_lists[year_id+1]
        tensor_diff_lists.append(list(torch.stack(next_year_tensor) - torch.stack(current_year_tensor)))
    return tensor_diff_lists


def generate_doc_vector_representation(year):
    tokenizer = RobertaTokenizer.from_pretrained(f"roberta-large")
    all_tokenized_sents = []
    max_len = 512
    with open(f"papers_{year}.txt") as f:
        lines = f.readlines()
        for line in tqdm(lines, total=len(lines), desc=f"load_{year}"):
            abstract = line.strip()
            sents = sent_tokenize(abstract)
            for sent in sents:
                tokenized_sent = tokenizer.tokenize(sent)
                if len(tokenized_sent) < max_len:
                    all_tokenized_sents.append(tokenized_sent)
    sent_vectors = []
    for sent_id, sent_tokens in tqdm(enumerate(all_tokenized_sents), total=len(all_tokenized_sents)):
        list_of_embedding = []
        for ind, token in enumerate(sent_tokens):
            token_id = roberta_tokenizer.get_vocab()[token]
            list_of_embedding.append(roberta_model.get_input_embeddings().weight[token_id].detach())
        sent_repr = sum(list_of_embedding) / len(sent_tokens)
        if torch.isinf(sent_repr).any().item():
            print(sent_id+":"+sent_tokens, flush=True)
        sent_vectors.append({"sent_tokens": sent_tokens, "sent_repr": sent_repr})
    vocab_sentences = {}
    count_token = {}
    for sent_tokens_vector in tqdm(sent_vectors, total=len(sent_vectors)):
        sent_tokens_t = deepcopy(sent_tokens_vector["sent_tokens"])
        sent_repr_t = deepcopy(sent_tokens_vector["sent_repr"])
        if torch.isinf(sent_repr_t).any().item():
            print(sent_tokens_t)
        token_set = list(set(sent_tokens_t))
        for token in token_set:
            if token in vocab_sentences:
                orginal_repr = deepcopy(vocab_sentences[token])
                vocab_sentences[token] += sent_repr_t
                if torch.isinf(vocab_sentences[token]).any().item():
                    print("plus_error:", sent_tokens_t)
                count_token[token] += 1
            else:
                vocab_sentences[token] = sent_repr_t
                if torch.isinf(vocab_sentences[token]).any().item():
                    print("equal_error:", sent_tokens_t)
                count_token[token] = 1
    vocab_representation = {}
    for token in vocab_sentences.keys():
        vocab_representation[token] = vocab_sentences[token] / count_token[token]
    torch.save(vocab_representation, f"vocab_representation_by_doc_{year}.th")
    return f"Finished year {year}!"


if __name__ == '__main__':
    # print(generate_doc_vector_representation(2000), flush=True)
    # exit(0)
    # sent_vectors = torch.load("doc_representation_org_2000.th")
    # for i in range(2000, 2022):
    #     print(generate_doc_vector_representation(i), flush=True)
    # exit(0)
    # year_vocab_repr = get_year_vocab_repr_tensor(prefix="vocab_representation_by_doc")
    # torch.save(year_vocab_repr, "year_vocab_by_doc_repr.pt")
    for i in range(2000, 2022):
        vocab_repr = torch.load(f"saved_model/vocab_representation_by_doc_{i}.th")
        print(i)
        for token in vocab_repr:
            if torch.isinf(vocab_repr[token]).any().item():
                print(token)
    # generate_doc_vector_representation(2001)
    # 2001, 2006, 2015
    exit(0)
    year_vocab_diff_repr2 = torch.load("year_vocab_by_doc_repr.pt")
    start = time.time()
    exit(0)
    year_vocab_repr2 = torch.load("year_vocab_repr.pt")
    year_vocab_diff_repr2 = get_year_vocab_repr_diff_tensor(year_vocab_repr2)
    torch.save(year_vocab_diff_repr2, "year_vocab_diff_repr.pt")
    end = time.time()
    ccc = end - start
    exit(0)
    year_vocab_repr_Tensor = torch.stack([torch.stack(year_vocab) for year_vocab in year_vocab_repr2])
    end2 = time.time()
    ccc2 = end2 - end
    set_list = []
    # for i in range(2001, 2022):
    #     vocab_repr = convert_to_id_based_representation(
    #         f"saved_model/vocab_representation/vocab_representation_{i}.th")
    #     vocab_repr_tensor, unused_vocab_repr_tensor = generate_vocab_representation_tensor(vocab_repr)
    #     exit(0)
    #     print(i, ":", len(vocab_repr))
    #     set_list.append(set(vocab_repr.keys()))
    # u = set.intersection(*set_list)
    # print(len(u))
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # print(len(tokenizer))
