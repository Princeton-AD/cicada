import argparse
import awkward as ak
import h5py
import numpy as np
import numpy.typing as npt
import os
import uproot

from dataclasses import dataclass
from pathlib import Path
from skimage.measure import block_reduce
from typing import List
from utils import IsReadableDir


@dataclass
class DataSource:
    et: ak.highlevel.Array
    ids: ak.highlevel.Array
    phi: ak.highlevel.Array
    eta: ak.highlevel.Array
    acceptanceFlag: ak.highlevel.Array
    size: int
    _calo_vars = ["iet", "ieta", "iphi"]
    _acceptance_vars = ["jetEta", "jetPt"]

    def __init__(self, in_file, tree_name, tree_gen):
        with uproot.open(in_file) as in_file:
            tree = in_file[tree_name]
            arrays = tree.arrays(self._calo_vars)
            eta = arrays["ieta"]
            phi = arrays["iphi"]
            et = arrays["iet"]
            self.size = len(eta)

            mask = (eta >= -28) & (eta <= 28)
            eta, phi, et = eta[mask], phi[mask], et[mask]
            eta = ak.where(eta < 0, eta, eta - 1)
            eta = eta + 28
            phi = (phi + 1) % 72

            ids = np.arange(len(eta))
            self.ids = ak.flatten(ak.broadcast_arrays(ids, eta)[0])
            self.phi = ak.flatten(phi, None)
            self.eta = ak.flatten(eta, None)
            self.et = ak.flatten(et, None)

            # Process Generator Information
            try:
                tree_gen = in_file[tree_gen]
                arrays = tree_gen.arrays(self._acceptance_vars)
                jetPT = arrays["jetPt"]
                jetEta = arrays["jetEta"]
                mask = (jetPT > 30.0) & (abs(jetEta) < 2.4)
                self.acceptanceFlag = ak.any(mask, axis=-1)
            except uproot.exceptions.KeyInFileError:
                self.acceptanceFlag = ak.Array([])

    def __len__(self):
        return self.size


def absoluteFilePaths(directory: Path, extension: str = "root"):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(extension):
                yield os.path.abspath(os.path.join(dirpath, f))


def get_split(events: int, split: npt.NDArray = np.array([0.6, 0.2, 0.2])):
    split = np.array(split)
    cumsum = np.cumsum(events * split).astype(int)
    cumsum = np.insert(cumsum, 0, 0)
    return [(i, j) for i, j in zip(cumsum, cumsum[1:])]


def convert(
    input_dir: Path, save_path: Path, calo_tree: str, acceptance_tree: str, split: bool
):
    create = True
    for in_ in absoluteFilePaths(input_dir):
        print("Processing {}".format(in_.split("/")[-1]))

        source = DataSource(in_, calo_tree, acceptance_tree)
        deposits = np.zeros((len(source), 72, 56))

        # Get raw data
        et = source.et.to_numpy()
        ids = source.ids.to_numpy()
        phi = source.phi.to_numpy()
        eta = source.eta.to_numpy()
        flags = source.acceptanceFlag.to_numpy()

        # Calculate regional deposits
        deposits[ids, phi, eta] = et

        # Reduce to towers
        region_et = block_reduce(deposits, (1, 4, 4), np.sum)
        region_et = np.where(region_et > 1023, 1023, region_et)

        # Save it in h5 file
        with h5py.File(save_path, "a") as h5f:
            if create:
                h5f.create_dataset(
                    "CaloRegions", data=region_et, maxshape=(None, 18, 14), chunks=True
                )
                h5f.create_dataset(
                    "AcceptanceFlag", data=flags, maxshape=(None,), chunks=True
                )
                create = False
                continue

            size = h5f["CaloRegions"].shape[0] + region_et.shape[0]
            h5f["CaloRegions"].resize((size), axis=0)
            h5f["CaloRegions"][-region_et.shape[0] :] = region_et
            if len(flags):
                h5f["AcceptanceFlag"].resize((size), axis=0)
                h5f["AcceptanceFlag"][-flags.shape[0] :] = flags

    if split:
        with h5py.File(save_path, "r") as h5f_in:
            cr = h5f_in["CaloRegions"]
            fs = h5f_in["AcceptanceFlag"]
            for part, (s, e) in enumerate(get_split(cr.shape[0])):
                output = "{}/{}_{}.h5".format(
                    save_path.parents[0], save_path.stem, part
                )
                with h5py.File(output, "w") as h5f_out:
                    h5f_out.create_dataset("CaloRegions", data=cr[s:e])
                    h5f_out.create_dataset("AcceptanceFlag", data=fs[s:e])


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Converts CMS Calorimeter Layer-1 Trigger with UCTRegions data from
           ROOT format to HDF5"""
    )
    parser.add_argument(
        "filepath", action=IsReadableDir, help="Input ROOT directory", type=Path
    )
    parser.add_argument("savepath", help="Output HDF5 files prefix", type=Path)
    parser.add_argument(
        "--calotree",
        help="Data tree",
        default="l1CaloTowerEmuTree/L1CaloTowerTree/L1CaloTower",
        type=str,
    )
    parser.add_argument(
        "--acceptance",
        help="Store acceptance flag",
        default="l1GeneratorTree/L1GenTree/Generator",
        type=str,
    )
    parser.add_argument(
        "--split", action="store_true", help="Split the dataset 60:20:20"
    )
    return parser.parse_args()


def main(args_in=None) -> None:
    args, config = parse_arguments()
    convert(args.filepath, args.savepath, args.calotree, args.acceptance, args.split)


if __name__ == "__main__":
    main()
