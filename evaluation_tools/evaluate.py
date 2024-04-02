from nltk.translate.meteor_score import single_meteor_score
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import argparse
import os
from nltk import word_tokenize

def normalized(abstract):
    tokens = word_tokenize(abstract)
    new_tokens = [token.lower() for token in tokens if len(token) > 4 and token.lower() not in stopwords]
    return new_tokens

def evaluate_single(gold_abstracts, pred_abstract):
    scores = []
    for gold_abstract in gold_abstracts:
        score = single_meteor_score(gold_abstract[0], pred_abstract[0])
        scores.append((pred_abstract[1], pred_abstract[0], gold_abstract[1], gold_abstract[0], score))
    return sorted(scores, key=lambda tup: tup[4], reverse=True)[0]


def evaluate_multiprocess(gold_abstracts, pred_abstracts, save_results=True, save_text=True,
                          results_filename=None, results_text_filename=None):
    pool = Pool(os.cpu_count())
    scores_list = tqdm(pool.imap(partial(evaluate_single, gold_abstracts), pred_abstracts), total=len(pred_abstracts))
    sorted_tuples = sorted(scores_list, key=lambda tup: tup[4], reverse=True)
    print(sorted_tuples[0])
    if save_results:
        with open(results_filename, "wb") as score_f:
            pickle.dump(sorted_tuples, score_f)
    if save_text:
        with open(results_text_filename, "w") as f:
            for pred_paper, pred_paper_tokens, gold_paper,  gold_paper_tokens, score in sorted_tuples:
                f.write(pred_paper + "\n")
                f.write("\n")
                f.write(" ".join(pred_paper_tokens) + "\n")
                f.write("\n")
                f.write(gold_paper + "\n")
                f.write("\n")
                f.write(" ".join(gold_paper_tokens) + "\n")
                f.write(str(score) + "\n")
                f.write("-" * 10 + "\n")
    return sorted_tuples, sorted_tuples[0], sorted_tuples[0][4]


def read_gold_data(gold_filename):
    gold_abstracts = []
    with open(gold_filename, "r") as gold_file:
        for line in gold_file:
            gold_abstracts.append(line.strip("\n"))
    return gold_abstracts


def read_gpt2_pred_data(pred_filename):
    pred_abstracts = []
    with open(pred_filename, "r") as pred_file:
        for line in pred_file:
            if line.startswith("<|endoftext|>"):
                pred_abstracts.append(line.strip("\n").split("<|endoftext|>")[1])
    return pred_abstracts


def normalized_abstracts(abstracts):
    new_abstracts = []
    for abstract in tqdm(abstracts, total=len(abstracts)):
        new_abstracts.append([normalized(abstract), abstract])
    return new_abstracts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate abstract.')
    parser.add_argument("--gold_abstracts", type=str, required=True,
                        help="gold abstracts file path")
    parser.add_argument("--pred_abstracts", type=str, required=True,
                        help="prediction abstracts file path")
    parser.add_argument("--output", type=str, required=True,
                        help="results output file path")
    parser.add_argument("--output_text", type=str, required=True,
                        help="results output text file path")
    args = parser.parse_args()
    golds = read_gold_data(args.gold_abstracts)
    preds = read_gpt2_pred_data(args.pred_abstracts)
    normalized_golds = normalized_abstracts(golds)
    normalized_preds = normalized_abstracts(preds)
    sorted_scores, max_score, max_score_value = \
        evaluate_multiprocess(normalized_golds, normalized_preds, save_results=True, save_text=True,
                              results_filename=args.output, results_text_filename=args.output_text)
    print(sorted_scores[0][0])
    print("\n")
    print(sorted_scores[0][1])
    print("\n")
    print(sorted_scores[0][2])
    print("\n")
    print(sorted_scores[0][3])
    print("\n")
    print(sorted_scores[0][4])
    # print(max_score_value)
