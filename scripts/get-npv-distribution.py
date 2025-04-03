import h5py
import pickle
import random
import hist
from hist import Hist
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_distribution_histogram(input_dir, axis):
    """
    Sample nPV across all datasets. Only creates nPV hist, omits ET
    Args:
        input_dir: Directory containing HDF5 files
        target_size: Desired size of the final dataset
    """
    # Process files in random order
    input_files = list(Path(input_dir).glob('*.h5'))
    # input_files = list(Path(input_dir).glob('*D*.h5'))
    random.shuffle(input_files)

    h = hist.Hist(axis, storage=hist.storage.Int64())
    for filepath in tqdm(input_files):
        print(f"Processing {filepath}")
        with h5py.File(filepath, 'r') as f:
            Y = f["nPV2"][:].astype("int16")
            h.fill(Y)

        print(h[:: hist.rebin(4)])
        print(f"sampled events: {sum(h)}")

    return h


# Usage example
if __name__ == "__main__":
    input_dir = "/hdfs/store/user/ligerlac/CICAD2025Training/"
    axis = hist.axis.Regular(75, 0.5, 75.5)

    h = get_distribution_histogram(input_dir, axis)
    print(h)

    with open("npv-dist.pkl", "wb") as f:
        pickle.dump(h, f)

    # with open("npv-dist.pkl", "rb") as f:
    #     h2 = pickle.load(f)

    h.plot(label=f"Total Number of Events: {sum(h)}")
    plt.gca().set_xlabel("Number of Primary Vertices")
    plt.legend()
    plt.savefig("npv-distribution.png")
    # h.plot1d(output="npv_distribution.png")
