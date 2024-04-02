import argparse
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analysis Results.')
    parser.add_argument("--results_file", type=str, required=True,
                        help="results output file path")
    parser.add_argument("--top", type=int, required=False, default=1,
                        help="top ith result")
    # parser.add_argument("--average", help="compute average", action="store_true")
    args = parser.parse_args()
    with open(args.results_file, "rb") as f:
        scores = pickle.load(f)
    sorted_tuples = sorted(scores, key=lambda tup: tup[4], reverse=True)

    # print(sum(sorted(scores)[-100:])/100)
    # print(sum([tup[2] for tup in sorted_tuples])/len(sorted_tuples))
    print(sorted_tuples[args.top-1][0])
    print("\n")
    print(sorted_tuples[args.top-1][1])
    print("\n")
    print(sorted_tuples[args.top-1][2])
    print("\n")
    print(sorted_tuples[args.top-1][3])
    print("\n")
    print(sorted_tuples[args.top-1][4])
    _, _, _, _, values = zip(*sorted_tuples)
    print("average:", sum(values) / len(values))