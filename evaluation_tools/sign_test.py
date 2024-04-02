import pickle
import os
import random
from collections import defaultdict
from math import comb


def compute_p_value(n, k):
    return comb(n, k)/(2**n)


def meteor_sign_test(output_dir="outputs2"):
    output_files = os.listdir(output_dir)
    scores_dict = defaultdict(list)
    for output_file in output_files:
        with open(output_dir + "/" + output_file, "rb") as f1:
            scores = pickle.load(f1)
        sorted_tuples = sorted(scores, key=lambda tup: tup[4], reverse=True)
        if "baseline" in output_file:
            for i in range(100):
                scores_dict["baseline"].append(sorted_tuples[i][4])
        else:
            approach_name = output_file.split("_")[6]
            for i in range(100):
                scores_dict[approach_name].append(sorted_tuples[i][4])
    baseline_scores = scores_dict["baseline"]
    sig_test_counter = defaultdict(int)
    for approach in scores_dict.keys():
        scores_list = scores_dict[approach]
        for score_1, score_2 in zip(baseline_scores, scores_list):
            if score_2 > score_1:
                sig_test_counter[approach] += 1
    return sig_test_counter


def ppl_sign_test(output_dir="outputs_ppl"):
    output_files = os.listdir(output_dir)
    scores_dict = defaultdict(list)
    for output_file in output_files:
        with open(output_dir + "/" + output_file, "rb") as f1:
            scores = pickle.load(f1)
        if "baseline" in output_file:
            scores_dict["baseline"] = scores
        else:
            approach_name = output_file.split("_")[6]
            scores_dict[approach_name] = scores
    baseline_scores = scores_dict["baseline"]
    sig_test_counter = defaultdict(int)
    for approach in scores_dict.keys():
        scores_list = scores_dict[approach]
        for score_1, score_2 in zip(baseline_scores, scores_list):
            if score_2 < score_1:
                sig_test_counter[approach] += 1
    return sig_test_counter


if __name__ == '__main__':
    print("kpm sign test.....\n")
    sig_test_counter = meteor_sign_test()
    for approach in sig_test_counter.keys():
        print(approach, ":", sig_test_counter[approach])

    print("\n....ppl sign test......\n")
    ppl_test_counter = ppl_sign_test()
    for approach in sig_test_counter.keys():
        print(approach, ":", sig_test_counter[approach])

    print("\n....kpp sign test......\n")
    kpp_test_counter = ppl_sign_test("outputs_kpp")
    for approach in sig_test_counter.keys():
        print(approach, ":", sig_test_counter[approach])