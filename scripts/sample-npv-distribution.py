import h5py
import numpy as np
from pathlib import Path
import random
from collections import defaultdict
import hist
from hist import Hist
from tqdm import tqdm


class DistributionSampler:
    def __init__(self, target_hist):
        """
        Initialize sampler with a target histogram that defines both binning and target distribution.
        
        Args:
            target_hist: hist.Hist instance defining the desired distribution and binning
        """
        self.target_hist = target_hist
        self.current_hist = hist.Hist(target_hist.axes[0], storage=hist.storage.Int64())
        
        self.sampled_x = []
        self.sampled_y = []
    

    def process_file(self, filepath):
        """Process a single HDF5 file using vectorized operations."""
        with h5py.File(filepath, 'r') as f:
            X = f["CaloRegions"][:].astype("int16")
            Y = f["nPV2"][:].astype("int16")
        
            # Calculate how many more samples we need in each bin
            remaining_hist = self.target_hist - self.current_hist
            
            # Create histogram of current file's Y values
            file_hist = hist.Hist(self.target_hist.axes[0], storage=hist.storage.Int64())
            file_hist.fill(Y)
            
            for bin_idx in range(len(remaining_hist.values())):
                needed_samples = remaining_hist.values()[bin_idx]
                if needed_samples <= 0:
                    continue
                    
                # Get indices where Y falls in this bin
                bin_edges = remaining_hist.axes[0].edges
                bin_mask = (Y >= bin_edges[bin_idx]) & (Y < bin_edges[bin_idx + 1])
                
                available_indices = np.where(bin_mask)[0]
                if len(available_indices) == 0:
                    continue
                    
                n_to_sample = min(len(available_indices), int(needed_samples))
                selected_indices = np.random.choice(
                    available_indices, n_to_sample, replace=False
                )
                
                self.sampled_x.extend(X[selected_indices])
                self.sampled_y.extend(Y[selected_indices])
                self.current_hist.fill(Y[selected_indices])

            print(self.current_hist[:: hist.rebin(4)])
            print(f"sampled events: {sum(self.current_hist)} / {sum(self.target_hist)}")

            # print(f"Maximum content: {h.values().max()}")  # prints: 8
            # print(f"Bin index with maximum: {np.argmax(h.values())}")  # prints: 3
            # print(f"Center of maximum bin: {h.axes[0].centers[np.argmax(h.values())]}")
            
            return (self.current_hist.values() >= self.target_hist.values()).all()


def create_dataset_with_distribution(input_dir, output_file, target_hist):
    """
    Create a new dataset matching the target distribution.
    
    Args:
        input_dir: Directory containing HDF5 files
        output_file: Path to save the output HDF5 file
        target_hist: hist.Hist instance defining desired distribution
    """
    sampler = DistributionSampler(target_hist)
    
    input_files = list(Path(input_dir).glob('*.h5'))
    random.shuffle(input_files)
    
    for filepath in input_files:
        print(f"Processing {filepath}")
        if sampler.process_file(filepath):
            break
    
    X_sampled = np.array(sampler.sampled_x)
    Y_sampled = np.array(sampler.sampled_y)
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('X', data=X_sampled)
        f.create_dataset('Y', data=Y_sampled)
        
    print(f"Saved {len(X_sampled)} samples to {output_file}")
    print("Target distribution:")
    print(target_hist)
    print("Achieved distribution:")
    print(sampler.current_hist)


# Usage example
if __name__ == "__main__":

    # From get-npv-distribution.py we know, that we can get at least 20,000 events per bin up to nPV = 65

    # h = hist.Hist(hist.axis.Regular(75, 0.5, 75.5), storage=hist.storage.Int64())
    h = hist.Hist(hist.axis.Regular(55, 10.5, 65.5), storage=hist.storage.Int64())

    # Example: Create uniform distribution
    h.values()[:] = 20000

    # Example: Gaussian distribution
    # x = np.arange(50) - 0.5  # bin centers
    # gaussian_values = np.exp(-0.5 * ((x - 25) / 5)**2)  # mean=25, std=5
    # gaussian_values = gaussian_values * (10000 / gaussian_values.sum())  # normalize to 10000 total entries
    # h.values()[:] = gaussian_values.astype(np.int64)
    
    # Example: triangular distribution
    # custom_hist = hist.Hist(hist.axis.Regular(50, -0.5, 49.5), storage=hist.storage.Int64())
    # counts = np.concatenate([np.arange(25), np.arange(25)[::-1]]) * 10
    # for bin_idx, count in enumerate(counts):
    #     custom_hist.fill(np.full(int(count), bin_idx))

    print(f"Attempting to create the following dist:\n{h}")

    input_dir = "/hdfs/store/user/ligerlac/CICAD2025Training/"
    output_file = "sampled_dataset.h5"

    create_dataset_with_distribution(input_dir, output_file, h)
