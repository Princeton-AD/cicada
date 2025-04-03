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
from tqdm import tqdm
from multiprocessing import Pool
from utils import IsReadableDir

@dataclass
class DataSource:
    et: ak.highlevel.Array
    ids: ak.highlevel.Array
    phi: ak.highlevel.Array
    eta: ak.highlevel.Array
    npv: ak.highlevel.Array
    run: ak.highlevel.Array
    acceptanceFlag: ak.highlevel.Array
    size: int
    _calo_vars = ["iet", "ieta", "iphi"]
    _acceptance_vars = ["jetEta", "jetPt"]
    _evt_vars = ["run", "lumi", "event"]

    def __init__(self, in_file, tree_name, tree_gen, tree_evt, tree_npv):
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

            # Process event Information
            if tree_evt is not None:
                tree_evt = in_file[tree_evt]
                arrays = tree_evt.arrays(self._evt_vars)
                self.run = ak.flatten(arrays["run"], None)
                self.lumi = ak.flatten(arrays["lumi"], None)
                self.event = ak.flatten(arrays["event"], None)
            else:
                self.run = ak.Array([])
                self.lumi = ak.Array([])
                self.event = ak.Array([])

            # Process NPV Information
            if tree_npv is not None:
                tree_npv = in_file[tree_npv]
                arrays = tree_npv.arrays(["nPV"])
                self.npv = ak.flatten(arrays["nPV"], None)
            else:
                self.npv = ak.Array([])


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


class FileHandler:
    def __init__(self, calo_tree: str, acceptance_tree: str, evt_tree: str, npv_tree: str):
        self.calo_tree = calo_tree
        self.acceptance_tree = acceptance_tree
        self.evt_tree = evt_tree
        self.npv_tree = npv_tree

    def get_array_dict(self, file_path: Path):
        tqdm.write(f"Processing {file_path.split('/')[-1]}")

        source = DataSource(file_path, self.calo_tree, self.acceptance_tree, self.evt_tree, self.npv_tree)

        deposits = np.zeros((len(source), 72, 56))

        et = source.et.to_numpy()
        ids = source.ids.to_numpy()
        phi = source.phi.to_numpy()
        eta = source.eta.to_numpy()
        flags = source.acceptanceFlag.to_numpy()
        run = source.run.to_numpy()
        lumi = source.lumi.to_numpy()
        event = source.event.to_numpy()
        npv = source.npv.to_numpy()

        # Calculate regional deposits
        deposits[ids, phi, eta] = et

        # Reduce to towers
        region_et = block_reduce(deposits, (1, 4, 4), np.sum)
        region_et = np.where(region_et > 1023, 1023, region_et)

        return {
            "region_et": region_et,
            "flags": flags,
            "run": run,
            "lumi": lumi,
            "event": event,
            "npv": npv
        }


def convert(
        input_dir: Path, save_path: Path, calo_tree: str, acceptance_tree: str, evt_tree: str, npv_tree: str, split: bool
):

    file_handler = FileHandler(calo_tree, acceptance_tree, evt_tree, npv_tree)

    file_paths = list(absoluteFilePaths(input_dir))
    with Pool(processes=32) as pool:
        data = list(tqdm(
            pool.imap(file_handler.get_array_dict, file_paths),
            total=len(file_paths)
        ))

    data = {k: np.concatenate([d[k] for d in data]) for k in data[0].keys()}

    with h5py.File(save_path, "x") as h5f:
        h5f.create_dataset(
            "CaloRegions", data=data["region_et"], maxshape=(None, 18, 14), chunks=True, dtype="int16"
        )
        h5f.create_dataset(
            "AcceptanceFlag", data=data["flags"], maxshape=(None,), chunks=True
        )
        h5f.create_dataset(
            "run", data=data["run"], maxshape=(None,), chunks=True,
        )
        h5f.create_dataset(
            "lumi", data=data["lumi"], maxshape=(None,), chunks=True
        )
        h5f.create_dataset(
            "event", data=data["event"], maxshape=(None,), chunks=True
        )
        h5f.create_dataset(
            "nPV", data=data["npv"], maxshape=(None,), chunks=True,
        )

    if split:
        with h5py.File(save_path, "r") as h5f_in:
            cr = h5f_in["CaloRegions"]
            fs = h5f_in["AcceptanceFlag"]
            pu = h5f_in["nPV"]
            rn = h5f_in["run"]
            for part, (s, e) in enumerate(get_split(cr.shape[0])):
                output = "{}/{}_{}.h5".format(
                    save_path.parents[0], save_path.stem, part
                )
                with h5py.File(output, "w") as h5f_out:
                    h5f_out.create_dataset("CaloRegions", data=cr[s:e])
                    h5f_out.create_dataset("AcceptanceFlag", data=fs[s:e])
                    h5f_out.create_dataset("nPV", data=pu[s:e])
                    h5f_out.create_dataset("run", data=rn[s:e])


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Converts CMS Calorimeter Layer-1 Trigger with UCTRegions data from
           ROOT format to HDF5"""
    )
    parser.add_argument(
        "filepath", action=IsReadableDir, help="Input ROOT directory", type=Path
    )
    parser.add_argument(
        "savepath", help="Output HDF5 file path", type=Path
    )
    parser.add_argument(
        "--calotree",
        help="Data tree",
        default="l1CaloTowerEmuTree/L1CaloTowerTree/L1CaloTower",
        type=str,
    )
    parser.add_argument(
        "--evttree",
        help="Event info tree, e.g. l1EventTree/L1EventTree/Event",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--npvtree",
        help="Tree w/ nPV info. E.g. data: npvNtuplizer/NPVTree, MC: l1EventTree/L1EventTree/Event",
        default=None,
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
    args = parse_arguments()
    convert(args.filepath, args.savepath, args.calotree, args.acceptance, args.evttree, args.npvtree, args.split)


if __name__ == "__main__":
    main()
