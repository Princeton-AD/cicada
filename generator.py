import h5py
import numpy as np
import numpy.typing as npt

from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import data
from typing import List, Tuple, Literal, Optional
import matplotlib.pyplot as plt
from utils import loss, quantize

class RegionETGenerator:
    def __init__(
        self, train_size: float = 0.5, val_size: float = 0.1, test_size: float = 0.4
    ):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = 42

    def get_generator(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        batch_size: int,
        drop_remainder: bool = False,
        weights: npt.NDArray = None,
    ) -> data.Dataset:
        if weights is None:
                dataset = data.Dataset.from_tensor_slices((X, y))
        else: 
                dataset = data.Dataset.from_tensor_slices((X, y, weights))            
        return (
            dataset.shuffle(210 * batch_size)
            .batch(batch_size, drop_remainder=drop_remainder)
            .repeat()
            .prefetch(data.AUTOTUNE)
        )

    def get_data(self, datasets_paths: List[Path]) -> npt.NDArray:
        inputs = []
        for dataset_path in datasets_paths:
            inputs.append(
                h5py.File(dataset_path, "r")["CaloRegions"][:].astype("float32")
            )
        X = np.concatenate(inputs)
        X = np.reshape(X, (-1, 18, 14, 1))
        return X

    def get_nPV(self, datasets_paths: List[Path]) -> npt.NDArray:
        inputs = []
        for dataset_path in datasets_paths:
            inputs.append(
                h5py.File(dataset_path, "r")["nPV"][:].astype("float32")
            )
        X = np.concatenate(inputs)
        return X

    def get_data_split(
        self, datasets_paths: List[Path]
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        X = self.get_data(datasets_paths)
        X_train, X_test = train_test_split(
            X, test_size=self.test_size, random_state=self.random_state
        )
        X_train, X_val = train_test_split(
            X_train,
            test_size=self.val_size / (self.val_size + self.train_size),
            random_state=self.random_state,
        )
        return (X_train, X_val, X_test)

    def get_benchmark(
        self, datasets: dict, filter_acceptance=True
    ) -> Tuple[dict, list]:
        signals = {}
        acceptance = []
        for dataset in datasets:
            if not dataset["use"]:
                continue
            signal_name = dataset["name"]
            for dataset_path in dataset["path"]:
                X = h5py.File(dataset_path, "r")["CaloRegions"][:].astype("float32")
                X = np.reshape(X, (-1, 18, 14, 1))
                try:
                    flags = h5py.File(dataset_path, "r")["AcceptanceFlag"][:].astype(
                        "bool"
                    )
                    fraction = 100.0#np.round(100 * sum(flags) / len(flags), 2)
                except KeyError:
                    fraction = 100.0
                if filter_acceptance:
                    X = X[flags]
                signals[signal_name] = X
                acceptance.append({"signal": signal_name, "acceptance": fraction})
        return signals, acceptance

    def generate_random_exposure_data(
        self,
        X_train: np.ndarray, 
        X_val: np.ndarray, 
        num_samples_train: int = 100000,
        num_samples_val: int = 100000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates two random numpy arrays with values in the range of the min and max
        values of the provided input arrays X_train and X_val.

        Args:
             X_train (np.ndarray): First input array.
             X_val (np.ndarray): Second input array.
             num_samples (int): Number of random samples to generate for each array.

        Returns:
             tuple:  Two numpy arrays of shape (num_samples, 18, 14, 1) filled with random values
                     in the range of the min and max of X_train and X_val.
        """
        # Find the combined min and max values from X_train and X_val
        global_min = 0#min(np.min(X_train), np.min(X_val))
        global_max = 1000#max(np.max(X_train), np.max(X_val))
        # Generate two random arrays with values in the global range
        rand_train = np.random.uniform(global_min, global_max, size=(num_samples_train, 18, 14, 1)).astype("float32")
        rand_val = np.random.uniform(global_min, global_max, size=(num_samples_val, 18, 14, 1)).astype("float32")
        return rand_train, rand_val

    def generate_random_exposure_data_from_hist(
        self,
        X_train: np.ndarray, 
        X_val: np.ndarray, 
        num_samples_train: int = 100000,
        num_samples_val: int = 100000,
        plot_hist: bool = True,
        hist_path: str = 'exposure_histogram.png'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates random numpy arrays by sampling from the 1d distribution of the
        cell values of the provided input arrays X_train and X_val.

        It creates a 1D histogram from the combined cell values of X_train and X_val. 
        It then samples from this distribution to generate new data, ensuring the 
        new data statistically resembles the original.

        Args:
                X_train (np.ndarray): First input array.
                X_val (np.ndarray): Second input array.
                num_samples_train (int): Number of random samples to generate for the training set.
                num_samples_val (int): Number of random samples to generate for the validation set.
                plot_hist (bool): If True, a plot of the histogram will be generated and saved.
                hist_path (str): The file path where the histogram plot will be saved.

        Returns:
                tuple: Two numpy arrays, rand_train and rand_val, of shapes 
                        (num_samples_train, 18, 14, 1) and (num_samples_val, 18, 14, 1) 
                        respectively, filled with values sampled from the input data's distribution.
        """
        # Combine and flatten the input data to get a 1D array of all cell values
        combined_data = np.concatenate((X_train.flatten(), X_val.flatten()))

        # Create histogram to sample from
        # np.histogram returns the frequency counts and the bin edges.
        counts, bin_edges = np.histogram(combined_data, bins=256, density=False)

        # The representative value is the center of the bin.
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Convert the frequency counts into a probability distribution.
        probabilities = counts / counts.sum()
        #print(bin_centers)
        #print(probabilities)

        # If requested, plot the histogram and save it
        if plot_hist:
                print(f"Plotting histogram and saving to {hist_path}...")
                plt.figure(figsize=(12, 7))
                # We use a bar plot to visualize the calculated histogram.
                # The width of the bars is set to the width of a single bin.
                bar_width = bin_edges[1] - bin_edges[0]
                plt.bar(bin_centers, counts, width=bar_width, align='center', edgecolor='black', alpha=0.8)
                
                plt.title("Histogram of Cell Values in Combined Trainign and Validation Data", fontsize=16)
                plt.xlabel("Cell Value", fontsize=12)
                plt.ylabel("Frequency (Number of Cells)", fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(hist_path)
                plt.close()
                print("Histogram saved.")

        # Generate random samples using the calculated distribution.
        print("Generating random samples for training set...")
        total_train_elements = num_samples_train * X_train.shape[1] * X_train.shape[2]
        random_samples_train_flat = np.random.choice(
                bin_centers,
                size=total_train_elements,
                p=probabilities
        )
        # Reshape the flat array to match X_train
        rand_train = random_samples_train_flat.reshape(
                (num_samples_train, X_train.shape[1], X_train.shape[2], 1)
        ).astype("float32")

        print("Generating random samples for validation set...")
        total_val_elements = num_samples_val * X_val.shape[1] * X_val.shape[2]
        random_samples_val_flat = np.random.choice(
                bin_centers,
                size=total_val_elements,
                p=probabilities
        )
        # Reshape the flat array to match X_val
        rand_val = random_samples_val_flat.reshape(
                (num_samples_val, X_val.shape[1], X_val.shape[2], 1)
        ).astype("float32")
        
        print("Data generation complete.")
        return rand_train, rand_val

    def generate_random_exposure_data_from_signal(
        self,    
        signal_train: np.ndarray,
        signal_val: np.ndarray,
        num_samples_train: int,
        num_samples_val: int,
        noise_level: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates noisy datasets by sampling from provided signal arrays.

        This function creates new training and validation arrays of a specified size by
        randomly drawing samples (with replacement) from the input arrays and adding
        Gaussian noise. The noise magnitude is determined by the standard deviation of
        the 'signal_train' data, scaled by the 'noise_level' factor.

        Args:
        signal_train (np.ndarray): The base signal array for training data.
                                        Expected shape: (n_samples, 18, 14, 1).
        signal_val (np.ndarray): The base signal array for validation data.
                                        Expected shape: (n_samples, 18, 14, 1).
        num_samples_train (int): The number of noisy samples to generate for the
                                        training set.
        num_samples_val (int): The number of noisy samples to generate for the
                                validation set.
        noise_level (float, optional): A factor to scale the noise. The noise's
                                        standard deviation will be this factor
                                        times the standard deviation of the
                                        'signal_train' data. Defaults to 0.1.

        Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two new numpy arrays:
                                        (noisy_train_data, noisy_val_data).
        """
        # Ensure the input arrays have samples to draw from
        if signal_train.shape[0] == 0 or signal_val.shape[0] == 0:
                raise ValueError("Input signal arrays cannot be empty.")

        # --- Generate Noisy Training Data ---

        # 1. Randomly select indices from the original training data (with replacement)
        print(signal_train.shape)
        train_indices = np.random.choice(signal_train.shape[0], size=num_samples_train, replace=True)
        print(train_indices.shape)
        # 2. Create the new base training set by picking the selected samples
        base_train = signal_train[train_indices]
        print(base_train.shape)

        # --- Generate Noisy Validation Data ---

        # 1. Randomly select indices from the original validation data (with replacement)
        val_indices = np.random.choice(signal_val.shape[0], size=num_samples_val, replace=True)
        # 2. Create the new base validation set by picking the selected samples
        base_val = signal_val[val_indices]

        # --- Create and Add Noise ---

        # Calculate noise scale based *only* on the training data to prevent data leakage
        data_std = np.std(signal_train)
        noise_std = data_std * noise_level

        # Generate Gaussian noise and add it to the new base sets
        noise_train = np.random.normal(loc=0.0, scale=noise_std, size=base_train.shape)
        print(noise_train.shape)
        noisy_train = (base_train + noise_train).astype("float32")

        noise_val = np.random.normal(loc=0.0, scale=noise_std, size=base_val.shape)
        noisy_val = (base_val + noise_val).astype("float32")

        return noisy_train, noisy_val        

    def generate_and_filter_anomalies_reproducible(
        self,
        normal_train: np.ndarray,
        anomaly_source: np.ndarray,
        pretrained_model: object,
        num_samples_to_generate: int,
        source_score_threshold: float = 180,
        final_score_threshold: float = 150,
        patch_size: int = 3,
        oversampling_factor: int = 3,
        batch_size: int = 1024
    ) -> np.ndarray:
        """
        Generates a reproducible set of synthetic anomalies using a random_state.
        """
        # Create a Random Number Generator from the seed 
        random_state = self.random_state
        rng = np.random.default_rng(random_state)

        # Filter the Anomaly Source 
        print("Filtering anomaly source")
        source_scores = pretrained_model.predict(anomaly_source)
        y= loss(anomaly_source, source_scores)
        y = quantize(np.log(y) * 32)
        high_quality_indices = np.where(y.flatten() >= source_score_threshold)[0]
        print(f"high quality anomaly sources={len(high_quality_indices)}")
        if len(high_quality_indices) == 0:
                raise ValueError(f"No source anomalies met the threshold of {source_score_threshold}.")
        high_quality_anomaly_source = anomaly_source[high_quality_indices]
        print(f"Using {len(high_quality_anomaly_source)} high-quality source anomalies.")

        # Generate and Filter Synthetic Candidates
        print("\n Generating and filtering synthetic anomalies")
        final_anomalies = []
        max_attempts = num_samples_to_generate * oversampling_factor
        generated_count = 0

        while len(final_anomalies) < num_samples_to_generate and generated_count < max_attempts:
                num_to_create_this_batch = min(batch_size, num_samples_to_generate - len(final_anomalies))
                
                #  Use the 'rng' object for all random operations 
                base_indices = rng.choice(normal_train.shape[0], size=num_to_create_this_batch, replace=True)
                candidate_batch = normal_train[base_indices].copy()
                generated_count += num_to_create_this_batch

                for i in range(num_to_create_this_batch):
                        grid = candidate_batch[i]
                        h, w, _ = grid.shape
                        
                        # Note: rng.integers is exclusive of the high value, so we don't need "- 1"
                        source_idx = rng.integers(0, high_quality_anomaly_source.shape[0])
                        anomaly_grid = high_quality_anomaly_source[source_idx]

                        max_start_y = h - patch_size
                        max_start_x = w - patch_size

                        # Note: high value is exclusive, so we add 1 to replicate randint's inclusive behavior
                        sy = rng.integers(0, max_start_y + 1)
                        sx = rng.integers(0, max_start_x + 1)
                        patch = anomaly_grid[sy:sy+patch_size, sx:sx+patch_size, :]
                        
                        dy = rng.integers(0, max_start_y + 1)
                        dx = rng.integers(0, max_start_x + 1)
                        grid[dy:dy+patch_size, dx:dx+patch_size, :] = patch
                        
                candidate_scores = pretrained_model.predict(candidate_batch)
                y= loss(candidate_batch, candidate_scores)
                y = quantize(np.log(y) * 32)
                passed_indices = np.where(y.flatten() >= final_score_threshold)[0]
                
                if len(passed_indices) > 0:
                        final_anomalies.append(candidate_batch[passed_indices])
                
                collected_count = len(np.vstack(final_anomalies)) if final_anomalies else 0
                print(f"Progress: {collected_count} / {num_samples_to_generate} collected...")

        if not final_anomalies:
                raise RuntimeError("Failed to generate any anomalies that met the final score threshold.")

        final_dataset = np.vstack(final_anomalies)
        return final_dataset[:num_samples_to_generate].astype("float32")        