import pickle
import os
import random
if __name__ == '__main__':
    output_files = os.listdir("outputs2")
    random.shuffle(output_files)
    inc = 1
    with open("secret_path_order.txt", "w") as f:
        for output_file in output_files:
            f.write(output_file+"\n")
            with open("outputs2/"+output_file, "rb") as f1:
                scores = pickle.load(f1)
            sorted_tuples = sorted(scores, key=lambda tup: tup[4], reverse=True)
            with open(str(inc) + ".out", "w") as f2:
                for i in range(7, 13):
                    f2.write(sorted_tuples[i][0] + "\n")
                    f2.write(str(sorted_tuples[i][1]) + "\n")
                    f2.write(sorted_tuples[i][2] + "\n")
                    f2.write(str(sorted_tuples[i][3]) + "\n")
                    f2.write(str(sorted_tuples[i][4]) + "\n")
                    f2.write("Error analysis: "+"\n")
                    f2.write("C1:Is topic clear and correct?"+"\n")
                    f2.write("C2:Is topic new?"+"\n")
                    f2.write("C3:Is problem clear and correct?"+"\n")
                    f2.write("C4:Is problem new?" + "\n")
                    f2.write("C5:Is method clear and correct?" + "\n")
                    f2.write("CC6: Is method new?" + "\n")
                    f2.write("\n")

            inc += 1



