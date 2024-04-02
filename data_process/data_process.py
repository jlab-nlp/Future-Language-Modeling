from tqdm import tqdm
import math
import requests
import shutil
import os
from sacrebleu.metrics import BLEU
from nltk.translate.meteor_score import meteor_score, single_meteor_score
from nltk import word_tokenize, sent_tokenize
import pickle
from nltk.util import ngrams
from collections import Counter
from multiprocessing import Pool
from functools import partial
from transformers import AutoTokenizer
from langdetect import detect
import langdetect
from math import log
from collections import defaultdict

stopwords = []
with open("stopwords.txt", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line.strip("\n")
        stopwords.append(line.split("\t")[1])




class Paper(object):
    def __init__(self, paper_id, title, year, abstract, url):
        self.paper_id = paper_id
        self.title = title
        self.year = year
        self.abstract = abstract
        self.url = url


def download_file(url, data_dir="data/temp"):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        with open(data_dir + "/" + local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return data_dir + "/" + local_filename


def read_bib(bib_file):
    bib_docs = []
    with open(bib_file, "r") as bib_tex:
        lines = bib_tex.readlines()
        bib_doc_temp = ""
        for line in tqdm(lines):
            if line.startswith("@"):
                if bib_doc_temp != "":
                    bib_docs.append(bib_doc_temp)
                bib_doc_temp = line
            else:
                bib_doc_temp += line
        bib_docs.append(bib_doc_temp)
    return bib_docs


def parse_bib_with_abstract(bib_docs):
    papers = []
    paper_id = 0
    for bib_doc in tqdm(bib_docs, total=len(bib_docs)):
        if "abstract =" in bib_doc:
            try:
                title = bib_doc.split("title = \"")[1].split("\"")[0]
                year = bib_doc.split("year = \"")[1].split("\"")[0]
                abstract = bib_doc.split("abstract = \"")[1].split("\"")[0]
                url = bib_doc.split("url = \"")[1].split("\"")[0]
                if abstract == "":
                    abstract = None
                print(1)
                print(abstract)
                papers.append(Paper(paper_id=paper_id, title=title, year=year, abstract=abstract, url=url))
                paper_id += 1
            except IndexError:
                title = bib_doc.split("title = \"")[1].split("\"")[0]
                year = bib_doc.split("year = \"")[1].split("\"")[0]
                abstract = bib_doc.split("abstract = {")[1].split("}")[0]
                url = bib_doc.split("url = \"")[1].split("\"")[0]
                papers.append(Paper(paper_id=paper_id, title=title, year=year, abstract=abstract, url=url))
                paper_id += 1
    return papers


def parse_bib(bib_docs):
    papers = []
    paper_id = 0
    for bib_doc in tqdm(bib_docs, total=len(bib_docs)):
        if "author =" not in bib_doc:
            continue
        if "abstract =" in bib_doc:
            try:
                title = bib_doc.split("title = \"")[1].split("\"")[0]
                year = bib_doc.split("year = \"")[1].split("\"")[0]
                abstract = bib_doc.split("abstract = \"")[1].split("\"")[0]
                url = bib_doc.split("url = \"")[1].split("\"")[0]
                if abstract == "":
                    abstract = None
                papers.append(Paper(paper_id=paper_id, title=title, year=year, abstract=abstract, url=url))
                paper_id += 1
            except IndexError:
                title = bib_doc.split("title = \"")[1].split("\"")[0]
                year = bib_doc.split("year = \"")[1].split("\"")[0]
                abstract = bib_doc.split("abstract = {")[1].split("}")[0]
                url = bib_doc.split("url = \"")[1].split("\"")[0]
                papers.append(Paper(paper_id=paper_id, title=title, year=year, abstract=abstract, url=url))
                paper_id += 1
        else:
            try:
                title = bib_doc.split("title = \"")[1].split("\"")[0]
                year = bib_doc.split("year = \"")[1].split("\"")[0]
                abstract = None
                url = bib_doc.split("url = \"")[1].split("\"")[0]
                papers.append(Paper(paper_id=paper_id, title=title, year=year, abstract=abstract, url=url))
                paper_id += 1
            except IndexError:
                title = bib_doc.split("title = {")[1].split("}")[0]
                year = bib_doc.split("year = \"")[1].split("\"")[0]
                abstract = None
                url = bib_doc.split("url = \"")[1].split("\"")[0]
                papers.append(Paper(paper_id=paper_id, title=title, year=year, abstract=abstract, url=url))
                paper_id += 1

    return papers


def papers_by_year(papers):
    year_papers_dict = {}
    for paper in papers:
        if paper.year not in year_papers_dict:
            year_papers_dict[paper.year] = [[paper.title, paper.abstract]]
        else:
            year_papers_dict[paper.year].append([paper.title, paper.abstract])
    return year_papers_dict


def crawl_and_parse_abstract(papers, is_write=False, output="papers.csv", error_link_out="error.out"):
    new_papers = []
    for paper in tqdm(papers, total=len(papers)):
        if paper.abstract is not None:
            new_papers.append(paper)
            if is_write:
                write_paper_to_csv(paper, output)
        else:
            downloaded_path = download_file(paper.url + ".pdf")
            os.system(f"./pdftotext -enc UTF-8 {downloaded_path} temp.txt")
            try:
                with open("temp.txt", "r") as temp:
                    lines = temp.readlines()
                    new_abstract = ""
                    for i, line in enumerate(lines):
                        if line.strip().lower() == "abstract":
                            abstract_line_index = i
                        try:
                            if "1" == line.split()[0]:
                                introduction_line_index = i
                                break
                        except IndexError:
                            pass
                    try:
                        for j in range(abstract_line_index + 1, introduction_line_index):
                            new_abstract += lines[j].strip() + " "
                    except IndexError:
                        print(paper.url)
                        write_error(paper.url)
                        os.system(f"rm -rf {downloaded_path}")
                        os.system(f"rm -rf temp.txt")
                        continue

                    # assert new_abstract is not None
                    if new_abstract == "" or new_abstract.strip() == "":
                        print(paper.url)
                        write_error(paper.url)
                        os.system(f"rm -rf {downloaded_path}")
                        os.system(f"rm -rf temp.txt")
                        continue
                    new_paper = Paper(paper_id=paper.paper_id, title=paper.title, year=paper.year,
                                      abstract=new_abstract, url=paper.url)
                    new_papers.append(new_paper)
                    if is_write:
                        write_paper_to_csv(new_paper, output)
            except FileNotFoundError:
                print(paper.url)
                write_error(paper.url)
            except UnicodeDecodeError:
                print(paper.url)
                write_error(paper.url)
            os.system(f"rm -rf {downloaded_path}")
            os.system(f"rm -rf temp.txt")
    return new_papers


def write_paper_to_csv(paper, output="papers.csv"):
    with open(output, "a+") as f:
        f.write(str(paper.paper_id) + "\t" + str(
            paper.year) + "\t" + paper.title + "\t" + paper.abstract + "\t" + paper.url + "\n")


def read_csv(input_file="papers.csv"):
    papers = []
    with open(input_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            paper_info = line.strip().split("\t")
            papers.append(Paper(paper_id=paper_info[0],
                                year=paper_info[1],
                                title=paper_info[2],
                                abstract=paper_info[3],
                                url=paper_info[4]))
    return papers


def write_error(error_link, error_link_out="error.out"):
    with open(error_link_out, "a+") as f:
        f.write(error_link + "\n")


def error_process(error_file="error.out"):
    error_urls = []
    with open(error_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            error_urls.append(line.strip())
    for error_url in error_urls:
        pass


def get_oracle_upper_bleu_bound_by_list(test_data, k=10):
    bleu = BLEU()
    scores = []
    for i, paper_i in tqdm(enumerate(test_data), total=len(test_data)):
        sys = []
        refs = []
        for j, paper_j in enumerate(test_data):
            if i == j:
                continue
            sys.append(paper_i[1])
            refs.append(paper_j[1])
        bleu_score = bleu.corpus_score(sys, refs)
        scores.append(bleu_score.score)
    return max(scores), sorted(scores)[-k:]


def get_oracle_upper_meteor_bound(test_data, k=10):
    scores = []
    for i, paper_i in tqdm(enumerate(test_data), total=len(test_data)):
        refs = []
        for j, paper_j in enumerate(test_data):
            if i == j:
                continue
            if paper_i[0] != paper_j[0]:
                refs.append(paper_j[1])
        score_temp = meteor_score(refs, paper_i[1])
        print(paper_i[0])
        print(score_temp)
        scores.append((paper_i[0], score_temp))
    with open("scores.pickle", "wb") as score_f:
        pickle.dump(scores, score_f)
    sorted_tuples = sorted(scores, key=lambda tup: tup[1])[-k:]
    return max(scores), sorted(scores, key=lambda tup: tup[1])[-k:], max(scores) / len(scores)


def get_oracle_upper_meteor_bound_pairwise(test_data, k=10):
    scores = []
    for i, paper_i in tqdm(enumerate(test_data), total=len(test_data)):
        refs = []
        for j, paper_j in enumerate(test_data):
            if i == j:
                continue
            if paper_i[0] != paper_j[0]:
                refs.append(paper_j[1])
                score_temp = single_meteor_score(paper_j[1], paper_i[1])
                scores.append((paper_i[1], paper_j[1], score_temp))
    with open("scores_pairwise.pickle", "wb") as score_f:
        pickle.dump(scores, score_f)
    sorted_tuples = sorted(scores, key=lambda tup: tup[2])[-k:]
    return sorted_tuples[-1], sorted_tuples[-k:]


def get_oracle_upper_meteor_bound_pairwise_single(test_data, paper_i):
    scores = []
    for j, paper_j in enumerate(test_data):
        if paper_i[0] != paper_j[0]:
            score_temp = single_meteor_score(paper_j[1], paper_i[1])
            scores.append((paper_i[1], paper_j[1], score_temp))
    return scores


def get_oracle_upper_meteor_bound_pairwise_multiprocess(test_data, k=10):
    pool = Pool(os.cpu_count())
    scores_list = list(tqdm(pool.imap(partial(get_oracle_upper_meteor_bound_pairwise_single, test_data), test_data),
                            total=len(test_data)))
    scores = [item for sublist in scores_list for item in sublist]

    sorted_tuples = sorted(scores, key=lambda tup: tup[2])[-100:]
    with open("scores_pairwise.pickle", "wb") as score_f:
        pickle.dump(sorted_tuples, score_f)
    return sorted_tuples[-1], sorted_tuples[-k:]


def get_oracle_upper_meteor_bound_pairwise_top_each_single(test_data, paper_i):
    max_score_tuples = []
    for j, paper_j in enumerate(test_data):
        if paper_i[0] != paper_j[0]:
            score_temp = single_meteor_score(paper_j[1], paper_i[1])
            max_score_tuples.append((paper_i[2], paper_i[1], paper_j[2], paper_j[1], score_temp))
    return sorted(max_score_tuples, key=lambda tup: tup[4], reverse=True)[0]


def get_oracle_upper_meteor_bound_pairwise_top_each_multiprocess(test_data, k=10):
    pool = Pool(os.cpu_count())
    scores_list = list(
        tqdm(pool.imap(partial(get_oracle_upper_meteor_bound_pairwise_top_each_single, test_data), test_data),
             total=len(test_data)))
    # scores = [item for sublist in scores_list for item in sublist]
    sorted_tuples = sorted(scores_list, key=lambda tup: tup[4], reverse=True)
    with open("scores-top-each.pickle", "wb") as score_f:
        pickle.dump(sorted_tuples, score_f)
    with open("scores-top-each.txt", "w") as f:
        for paper_i, paper_i_tokens, paper_j, paper_j_tokens, score in sorted_tuples:
            f.write(paper_i + "\n")
            f.write("\n")
            f.write(" ".join(paper_i_tokens)+"\n")
            f.write("\n")
            f.write(paper_j + "\n")
            f.write("\n")
            f.write(" ".join(paper_j_tokens) + "\n")
            f.write(str(score) + "\n")
            f.write("-" * 10 + "\n")
    return sorted_tuples[0], sorted_tuples[:k], sorted_tuples


def get_oracle_upper_bleu_bound_by_element(test_data, k=10):
    bleu = BLEU()
    scores = []
    max_bscore = float("-inf")
    for i, paper_i in tqdm(enumerate(test_data), total=len(test_data)):
        for j, paper_j in enumerate(test_data):
            if i == j:
                continue
            bleu_score = bleu.corpus_score(paper_i[1], paper_j[1])
            scores.append(bleu_score.score)
            if bleu_score.score > max_bscore:
                max_bscore = bleu_score.score
                print(max_bscore)

    return max(scores), sorted(scores)[-k:]


def check_overlap(test_data):
    pair_computed = []
    scores = []
    for i, paper_i in tqdm(enumerate(test_data), total=len(test_data)):
        for j, paper_j in enumerate(test_data):
            if i == j:
                continue
            pair_string = str(i) + "_" + str(j)
            if pair_string not in pair_computed and paper_i[0] != paper_j[0]:
                score_temp = meteor_score([paper_j[1]], paper_i[1])
                pair_computed.append(pair_string)
                scores.append((paper_j[0], paper_j[0], score_temp))
                if score_temp > 0.9:
                    print("\n" + pair_string, "\n", paper_j[0], "\n", paper_j[0] + "\n")
    with open("scores_dict.pickle", "wb") as score_f:
        pickle.dump(scores, score_f)
    return scores


def check_overlap_by_equal(test_data):
    for i, paper_i in tqdm(enumerate(test_data), total=len(test_data)):
        for j, paper_j in enumerate(test_data):
            if i == j:
                continue
            else:
                if paper_i[0] == paper_j[0] or paper_i[1] == paper_j[1]:
                    print(paper_i[0] + "|" + paper_j[0])


def output_abstract_line_by_line(papers, start_year, end_year):
    year_papers_dict = papers_by_year(papers)
    if start_year == end_year:
        save_file = "papers_" + str(start_year) + ".txt"
        year_set = year_papers_dict[str(start_year)]
        with open(save_file, "w") as f:
            for i, paper in tqdm(enumerate(year_set), total=len(year_set)):
                f.write(paper[1] + "\n")
    else:
        save_file = "papers_" + str(start_year) + "_" + str(end_year) + ".txt"
        with open(save_file, "w") as f:
            for i in range(start_year, end_year + 1):
                if str(i) not in year_papers_dict:
                    continue
                year_set = year_papers_dict[str(i)]
                for i, paper in tqdm(enumerate(year_set), total=len(year_set)):
                    f.write(paper[1] + "\n")


def generate_previous_training_set(papers, start_year, final_year):
    for i in range(start_year, final_year + 1):
        output_abstract_line_by_line(papers, i, final_year)


def get_unigram_freq(papers):
    frequencies = Counter([])
    for i, paper in tqdm(enumerate(papers), total=len(papers), desc="calculating unigram..."):
        unigram = ngrams(word_tokenize(paper.abstract.lower()), 1)
        frequencies += Counter(unigram)
    with open("unigram_frequency.txt", "w") as f:
        for i, token_frequency in tqdm(enumerate(frequencies.most_common()), total=len(frequencies), desc="saving..."):
            token = token_frequency[0][0]
            frequency = token_frequency[1]
            f.write(str(i + 1) + "\t" + token + "\t" + str(frequency) + "\n")
    return frequencies


def get_ngram_freq(papers, num_grams=2):
    if num_grams == 1:
        return get_unigram_freq(papers)
    frequencies = Counter([])
    for i, paper in tqdm(enumerate(papers), total=len(papers), desc=f"calculating {num_grams}-gram..."):
        bigram = ngrams(word_tokenize(paper.abstract.lower()), num_grams)
        frequencies += Counter(bigram)
    with open(str(num_grams) + "-gram_frequency.txt", "w") as f:
        for i, token_frequency in tqdm(enumerate(frequencies.most_common()), total=len(frequencies), desc="saving..."):
            bigram = " ".join(list(token_frequency[0]))
            frequency = token_frequency[1]
            f.write(str(i + 1) + "\t" + bigram + "\t" + str(frequency) + "\n")
    return frequencies


tokenizer_kwargs = {'cache_dir': None, 'use_fast': True, 'revision': 'main', 'use_auth_token': None}
tokenizer = AutoTokenizer.from_pretrained("gpt2", **tokenizer_kwargs)


def get_unigram_freq_by_year(papers, year, save_dir="ngrambyyear", cut_year=2000, tokenize_fn=word_tokenize,
                             start_year=1952):
    year_papers_dict = papers_by_year(papers)
    if year <= cut_year:
        if year < cut_year:
            print(f"The year smaller than {cut_year} will be considered in 2010.")
            return None
        year_papers = []
        for i in range(start_year, cut_year + 1):
            if str(i) not in year_papers_dict:
                continue
            else:
                year_papers.extend(year_papers_dict[str(i)])
    else:
        year_papers = year_papers_dict[str(year)]
    # year_papers = year_papers_dict[str(year)]
    frequencies = Counter([])
    for i, paper in tqdm(enumerate(year_papers), total=len(year_papers),
                         desc=f"Computing unigram freq for year {year}"):
        unigram = ngrams(tokenize_fn(paper[1].lower()), 1)
        frequencies += Counter(unigram)
    with open(os.path.join(save_dir,
                           "unigram_frequency-" + str(year)
                           + "-cut-" + str(cut_year) + "-start-" + str(start_year) + "-tokenizer-" + tokenize_fn.__name__ + ".txt"), "w") as f:
        for i, token_frequency in tqdm(enumerate(frequencies.most_common()), total=len(frequencies), desc="saving..."):
            ngram = token_frequency[0][0]
            frequency = token_frequency[1]
            f.write(str(i + 1) + "\t" + ngram + "\t" + str(frequency) + "\n")
    return frequencies


def get_ngram_freq_by_year(papers, year, num_grams=2, save_dir="ngrambyyear", cut_year=2000, tokenize_fn=word_tokenize,
                           start_year=1952):
    if num_grams == 1:
        return get_unigram_freq_by_year(papers, year, cut_year=cut_year, tokenize_fn=tokenize_fn, start_year=start_year)
    year_papers_dict = papers_by_year(papers)
    if year <= cut_year:
        if year < cut_year:
            print(f"The year smaller than {cut_year} will be considered in 2010.")
            return None
        year_papers = []
        for i in range(start_year, cut_year + 1):
            if str(i) not in year_papers_dict:
                continue
            else:
                year_papers.extend(year_papers_dict[str(i)])
    else:
        year_papers = year_papers_dict[str(year)]
    frequencies = Counter([])
    for i, paper in tqdm(enumerate(year_papers), total=len(year_papers),
                         desc=f"Computing{num_grams}-gram freq for year {year}"):
        ngram = ngrams(tokenizer.tokenize(paper[1].lower()), 2)
        frequencies += Counter(ngram)
    with open(os.path.join(save_dir,
                           str(num_grams) + "-gram_frequency-" + str(year)
                           + "-cut-" + str(cut_year) + "-start-" + str(start_year) +
                           "-tokenizer-" + tokenize_fn.__name__ + ".txt"), "w") as f:
        for i, token_frequency in tqdm(enumerate(frequencies.most_common()), total=len(frequencies), desc="saving..."):
            ngram = " ".join(list(token_frequency[0]))
            frequency = token_frequency[1]
            f.write(str(i + 1) + "\t" + ngram + "\t" + str(frequency) + "\n")
    return frequencies


def get_ngram_freq_by_year_range(papers, start_year, end_year, ngram_range, cut_year=2000, tokenize_fn=word_tokenize,
                                 inner_start_year=1952):
    for i in range(ngram_range):
        gram = i + 1
        for year in range(start_year, end_year + 1):
            get_ngram_freq_by_year(papers, year, num_grams=gram, cut_year=cut_year, tokenize_fn=tokenize_fn,
                                   start_year=inner_start_year)


def filter_out(papers, output="papers-filtered-final-without-semeval.csv"):
    # Filter out workshop shared task papers because it is not related to real trends
    # Filter out abstracts in other languages.
    # Filter out abstracts that have encoding issues.
    filtered_papers = []
    for paper in tqdm(papers, total=len(papers)):
        try:
            if detect(paper.abstract) == 'en' \
                    and "shared task" not in paper.abstract.lower() \
                    and "shared tasks" not in paper.abstract.lower() \
                    and "semeval" not in paper.abstract.lower() \
                    and "sem-eval" not in paper.abstract.lower() \
                    and "{s}em{e}val" not in paper.abstract.lower() \
                    and "{s}em{e}val" not in paper.title.lower() \
                    and "semeval" not in paper.url.lower() \
                    and (len(paper.abstract.split(" ")) > 10):
                flag = True
                for token in word_tokenize(paper.abstract):
                    try:
                        if detect(token) == 'ro':
                            flag = False
                            break
                    except langdetect.lang_detect_exception.LangDetectException:
                        continue
                if flag:
                    filtered_papers.append(paper)
                    write_paper_to_csv(paper, output=output)
        except langdetect.lang_detect_exception.LangDetectException:
            print("No features in text.")
            continue
    return filtered_papers


def get_year_span_paper_num(papers, start_year, end_year):
    year_num_dict = {}
    year_papers_dict = papers_by_year(papers)
    sentence_year_num_dict = {}
    token_year_num_dict = {}
    for key in tqdm(year_papers_dict.keys(), total=len(year_papers_dict.keys())):
        if int(key) < start_year or int(key) > end_year:
            continue
        year_num_dict[key] = len(year_papers_dict[key])
        for title, abstract in year_papers_dict[key]:
            if key not in sentence_year_num_dict:
                sentence_year_num_dict[key] = len(sent_tokenize(abstract))
                token_year_num_dict[key] = len(word_tokenize(abstract))
            else:
                sentence_year_num_dict[key] += len(sent_tokenize(abstract))
                token_year_num_dict[key] += len(word_tokenize(abstract))
    num_papers = 0
    num_of_sentences = 0
    num_of_tokens = 0
    for i in range(start_year, end_year + 1):
        if str(i) in year_num_dict.keys():
            num_papers += year_num_dict[str(i)]
            num_of_sentences += sentence_year_num_dict[str(i)]
            num_of_tokens += token_year_num_dict[str(i)]
    return num_papers, num_of_sentences*1.0/num_papers, num_of_tokens*1.0/num_papers


def normalized_dataset(data_set):
    new_dataset = []
    for i, paper_i in tqdm(enumerate(data_set), total=len(data_set)):
        new_dataset.append([paper_i[0], normalized(paper_i[1]), paper_i[1]])
    return new_dataset


def normalized(abstract):
    tokens = word_tokenize(abstract)
    new_tokens = [token.lower() for token in tokens if len(token) > 4 and token.lower() not in stopwords]
    return new_tokens


# TODO:initiaze the year based data
def generate_abstract_year_data(papers, start_year=1952, cut_year=2000, end_year=2019):
    with open(f"papers_with_year_start_{start_year}_cut_{cut_year}_end_{end_year}.csv", "w") as f:
        f.write("previous_year_index"+"\t"+"text"+"\n")
        for paper in tqdm(papers, total=len(papers)):
            if start_year <= int(paper.year) <= end_year:
                if int(paper.year) <= cut_year:
                    f.write(str(-1)+"\t"+paper.abstract+"\n")
                else:
                    f.write(str(int(paper.year) - 1 - cut_year) + "\t" + paper.abstract + "\n")


def get_ngram_dict_by_year(start_year=1952, cut_year=2000, end_year=2019):
    year_unigram_dict = defaultdict(lambda: defaultdict(float))
    year_bigram_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    year_trigram_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for year in range(start_year, end_year + 1, 1):
        unigram_file = f"ngrambyyear/unigram_frequency-{year}-cut-{cut_year}-tokenizer-tokenize.txt"
        bigram_file = f"ngrambyyear/2-gram_frequency-{year}-cut-{cut_year}-tokenizer-tokenize.txt"
        trigram_file = f"ngrambyyear/3-gram_frequency-{year}-cut-{cut_year}-tokenizer-tokenize.txt"
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
        for token in unigram_freq_dict.keys():
            unigram_freq_prob_dict[token] = log(unigram_freq_dict[token] / total)
        year_unigram_dict[year] = unigram_freq_prob_dict

        bigram_freq_dict = defaultdict(lambda: defaultdict(int))
        with open(bigram_file) as f:
            lines = f.readlines()
            for line in lines:
                token_freq = line.strip("\n").split("\t")
                tokens = token_freq[1].split(" ")
                freq = int(token_freq[2])
                bigram_freq_dict[tokens[0]][tokens[1]] = freq
        bigram_freq_prob_dict = defaultdict(lambda: defaultdict(float))
        for token in bigram_freq_dict.keys():
            for token2 in bigram_freq_dict[token].keys():
                bigram_freq_prob_dict[token][token2] = log(bigram_freq_dict[token][token2] / unigram_freq_dict[token])
        year_bigram_dict[year] = bigram_freq_prob_dict

        trigram_freq_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        total = 0
        count = 0
        with open(trigram_file) as f:
            lines = f.readlines()
            for line in lines:
                token_freq = line.strip("\n").split("\t")
                tokens = token_freq[1].split(" ")
                freq = int(token_freq[2])
                total += freq
                trigram_freq_dict[tokens[0]][tokens[1]][tokens[2]] = freq
                count += 1
        trigram_freq_prob_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for token in trigram_freq_dict.keys():
            for token2 in trigram_freq_dict[token].keys():
                for token3 in trigram_freq_dict[token2].keys():
                    trigram_freq_prob_dict[token][token2][token3] = log(trigram_freq_dict[token][token2][token3]
                                                                        / bigram_freq_dict[token][token2])
        year_trigram_dict[year] = trigram_freq_prob_dict
    return year_unigram_dict, year_bigram_dict, year_trigram_dict


def compute_ngram_probability_in_context(sentence, year_unigram_dict, year_bigram_dict, year_trigram_dict,
                                         year, l1, l2, l3, cut_year=2000):
    unigram_dict = year_unigram_dict[year]
    bigram_dict = year_bigram_dict[year]
    trigram_dcit = year_trigram_dict[year]
    tokens = tokenizer.tokenize(sentence)
    tokens_prob = []
    for i, token in enumerate(tokens):
        if i == 0:
            tokens_prob.append(l1*unigram_dict[token])
        elif i == 1:
            tokens_prob.append(l1*unigram_dict[token]+l2*bigram_dict[tokens[i-1]][token])
        else:
            tokens_prob.append(l1*unigram_dict[token] + l2*bigram_dict[tokens[i - 1]][token]
                               + l3*trigram_dcit[tokens[i - 2]][[tokens[i - 1]][token]])
    return tokens_prob


# Use EM algorithm to find l1, l2, l3


def generate_abstract_year_ngram_prob_data(papers, start_year=1952, cut_year=2000, end_year=2019):
    with open(f"papers_ngram_with_year_start_{start_year}_cut_{cut_year}_end_{end_year}.csv", "w") as f:
        f.write("previous_year_index"+"\t"+"text"+"\n"+"ngram_prob")
        for paper in tqdm(papers, total=len(papers)):
            if start_year <= int(paper.year) <= end_year:
                if int(paper.year) <= cut_year:
                    f.write(str(-1)+"\t"+paper.abstract+"\n")
                else:
                    f.write(str(int(paper.year) - 1 - cut_year) + "\t" + paper.abstract + "\n")


if __name__ == '__main__':

    # print(tokenizer.tokenize("I love you."))
    # print(str(tokenizer.tokenize.__name__))
    # exit(0)
    # bib_docs = read_bib("data/anthology+abstracts.bib")
    # papers = parse_bib(bib_docs)
    # new_papers = crawl_and_parse_abstract(papers, is_write=True)

    # papers = read_csv()


    papers = read_csv(input_file="papers-filtered-final-without-semeval.csv")
    # print(len(papers))
    # new_papers = filter_out(papers)
    # print(len(new_papers))
    # exit(0)
    year_papers_dict = papers_by_year(papers)

    # for key in year_papers_dict.keys():
    #     print(len(year_papers_dict[key]))
    print(get_year_span_paper_num(papers, 2003, 2019))
    print(len(papers))
    exit(0)
    get_ngram_freq(papers, num_grams=1)
    for i in range(2000, 2020):
        output_abstract_line_by_line(papers, start_year=i, end_year=i)
    exit(0)
    generate_abstract_year_data(papers, start_year=2003, cut_year=2000, end_year=2019)
    # generate_abstract_year_data(papers, start_year=1952, cut_year=2000, end_year=2020)
    # generate_abstract_year_data(papers, start_year=2020, cut_year=2000, end_year=2020)
    # generate_abstract_year_data(papers, start_year=2021, cut_year=2000, end_year=2021)
    exit(0)
    # get_ngram_freq(papers, num_grams=2)
    # get_ngram_freq(papers, num_grams=3)
    # generate_previous_training_set(papers, 1952, 2019)
    # output_abstract_line_by_line(papers, start_year=2017, end_year=2019)
    output_abstract_line_by_line(papers, start_year=1952, end_year=2020)

    output_abstract_line_by_line(papers, start_year=2020, end_year=2020)
    output_abstract_line_by_line(papers, start_year=2021, end_year=2021)
    exit(0)

    # print(get_year_span_paper_num(papers, 1952, 1988))
    # # exit(0)
    # start_year = 2019
    # # tokenizer.tokenize
    # # # used to generate stopwords
    # # get_ngram_freq_by_year_range(papers, start_year, 2019, 1, cut_year=2019, tokenize_fn=word_tokenize)
    # # used to generate the training ngram
    get_ngram_freq_by_year_range(papers, 2019, 2019, 1, cut_year=2019, tokenize_fn=tokenizer.tokenize,
                                 inner_start_year=2017)
    exit(0)
    # get_ngram_freq_by_year_range(papers, 2020, 2020, 3, cut_year=2020, tokenize_fn=tokenizer.tokenize)
    # get_ngram_freq_by_year_range(papers, 2000, 2019, 3, cut_year=2000, tokenize_fn=tokenizer.tokenize)
    # exit(0)
    test_set = year_papers_dict["2021"]
    normalized_test_set = normalized_dataset(test_set)
    max_score, top_10, all_scores = get_oracle_upper_meteor_bound_pairwise_top_each_multiprocess(normalized_test_set)
    print(len(all_scores))
    print(max_score)
    print(top_10)

    # with open("scores_pairwise.pickle", "rb") as f:
    #     scores = pickle.load(f)
    # print(len(scores))
    # print(sum(scores)/len(scores))
    # import numpy as np
    # print(np.median(scores))
    # print(min(scores))
    # exit(0)
    # papers_with_abstract = parse_bib_with_abstract(bib_docs)
    # year_papers_dict_with_abstract = papers_by_year(papers_with_abstract)
    # for key in year_papers_dict_with_abstract.keys():
    #     print(key, len(year_papers_dict_with_abstract[key]))
