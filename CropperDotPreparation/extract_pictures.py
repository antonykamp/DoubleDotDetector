import numpy as np
import matplotlib.pyplot as plt
import progressbar
import sys
import os

# sys.argv
# 1 input path
# 2 output path npy
# 3 output path png
# (4 input path label array)
# (5 input path label searched for)


def save_picture(data, output_path):
    img = plt.imshow(data, cmap='bwr')
    plt.savefig(output_path, pad_inches=0, bbox_inches='tight')
    plt.clf()
    

def save_npy(data, output_path):
    np.save(output_path, data, allow_pickle=True)
    
    
def main():
    input_path = sys.argv[1]
    output_path_png = sys.argv[2]
    output_path_csv = sys.argv[3]

    data = np.load(input_path, allow_pickle=True)

    if len(sys.argv) < 5:
        # input doesn't use extern label array
        label = np.array(len(data)*[1])
        check = 1
    else:
        # input uses extern label array
        label = np.load(sys.argv[4], allow_pickle=True)
        check = int(sys.argv[5])

    name = os.path.basename(sys.argv[1]).split(".")[0]
    plt.axis('off')
    bar = progressbar.ProgressBar()
    for i in bar(range(len(data))):
        if label[i] == check:
            d = data[i]
            save_picture(d, os.path.join(output_path_png, "{}-{}".format(name, i)))
            save_npy(d, os.path.join(output_path_csv, "{}-{}".format(name, i)))


if __name__ == "__main__":
    main()