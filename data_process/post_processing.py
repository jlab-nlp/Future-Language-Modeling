

def read_gpt2_pred_data(pred_filename):
    pred_abstracts = []
    with open(pred_filename, "r") as pred_file:
        for line in pred_file:
            if line.startswith("<|endoftext|>"):
                pred_abstracts.append(line.strip("\n").split("<|endoftext|>")[1])
    return pred_abstracts

if __name__ == '__main__':
    pred_abstracts = read_gpt2_pred_data("")
    with open("",
              "w") as f:
        for pred_abstract in pred_abstracts:
            texts = pred_abstract.split("  ")
            text = texts[0]
            for i in range(len(texts)):
                if len(texts[i]) > 90:
                    f.write("=== GENERATED SEQUENCE 1 ===\n")
                    f.write("<|endoftext|> " + texts[i] + "\n")
                    break


            # if len(text) <= len("of the Art in Natural Language Processing  Computational Linguistics  Volume 33, Number 1  "):
            #
            #     # pred_text = texts[1]
            #     f.write("=== GENERATED SEQUENCE 1 ===\n")
            #     f.write("<|endoftext|> " + pred_text + "\n")
            # else:
            #     pred_text = texts[0]
            #     f.write("=== GENERATED SEQUENCE 1 ===\n")
            #     f.write("<|endoftext|>" + pred_text + "\n")

