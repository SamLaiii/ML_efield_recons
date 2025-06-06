import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import argparse
import logging
import numpy as np
from tqdm import tqdm
import grand.dataio.root_trees as rt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

@dataclass
class DataDirs:
    swap: Path
    nonswap: Path

@dataclass
class SplitDirs:
    NJ: DataDirs
    AN: DataDirs

# Directory configuration for train, validation, and test
DATA_CONFIG: Dict[str, SplitDirs] = {
    "train": SplitDirs(
        NJ=DataDirs(Path("/home/923714256/new_dc2/DC2RF2_NJ/Train"), Path("/home/923714256/new_dc2/DC2RF2_NJ/Train/non_swap")),
        AN=DataDirs(Path("/home/923714256/new_dc2/DC2RF2_AN/Train"), Path("/home/923714256/new_dc2/DC2RF2_AN/Train/non_swap")),
    ),
    "validation": SplitDirs(
        NJ=DataDirs(Path("/home/923714256/new_dc2/DC2RF2_NJ/Validation"), Path("/home/923714256/new_dc2/DC2RF2_NJ/Validation/non_swap")),
        AN=DataDirs(Path("/home/923714256/new_dc2/DC2RF2_AN/Validation"), Path("/home/923714256/new_dc2/DC2RF2_AN/Validation/non_swap")),
    ),
    "test": SplitDirs(
        NJ=DataDirs(Path("/home/923714256/new_dc2/DC2RF2_NJ/Test"), Path("/home/923714256/new_dc2/DC2RF2_NJ/Test/non_swap")),
        AN=DataDirs(Path("/home/923714256/new_dc2/DC2RF2_AN/Test"), Path("/home/923714256/new_dc2/DC2RF2_AN/Test/non_swap")),
    ),
}


def collect_root_files(directory: Path, pattern: str) -> List[Path]:
    """Return sorted list of files matching a glob pattern in the given directory."""
    return sorted(directory.glob(pattern))


def get_matching_files_for_split(
    split_dirs: SplitDirs,
    split_name: str,
) -> Tuple[List[Tuple[Path, Path, Path, Path]], List[Tuple[Path, Path, Path, Path]]]:
    """
    For a given split (train/validation/test), collect matching swap and non-swap file quadruplets:
    (adc_NJ, shower_NJ, adc_AN, shower_AN).
    """
    patterns = {
        "adc": "adc_*_L1_0000.root",
        "shower": "shower_*_L0_0000.root",
    }
    quads = {"swap": [], "nonswap": []}
    for mode in quads:
        adc_NJ = collect_root_files(getattr(split_dirs.NJ, mode), patterns["adc"])
        sd_NJ  = collect_root_files(getattr(split_dirs.NJ, mode), patterns["shower"])
        adc_AN = collect_root_files(getattr(split_dirs.AN, mode), patterns["adc"])
        sd_AN  = collect_root_files(getattr(split_dirs.AN, mode), patterns["shower"])
        assert len(adc_NJ) == len(sd_NJ) == len(adc_AN) == len(sd_AN), (
            f"Mismatch in counts for {split_name} {mode}: "
            f"ADC_NJ={len(adc_NJ)}, SD_NJ={len(sd_NJ)}, ADC_AN={len(adc_AN)}, SD_AN={len(sd_AN)}"
        )
        quads[mode] = list(zip(adc_NJ, sd_NJ, adc_AN, sd_AN))
        logger.info(f"{split_name} [{mode}]: found {len(adc_NJ)} files")
    return quads["swap"], quads["nonswap"]


def maybe_slice_swap_traces(
    x_clean: np.ndarray,
    y_clean: np.ndarray,
    z_clean: np.ndarray,
    x_noisy: np.ndarray,
    y_noisy: np.ndarray,
    z_noisy: np.ndarray,
    *,
    swap_prob: float = 0.5,
    target_start: int = 300,
    target_end:   int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    With probability swap_prob, swap a slice [target_start:target_end]
    between clean and noisy triplets along a random source region.
    Returns the (x_clean, x_noisy, y_clean, y_noisy, z_clean, z_noisy).
    """
    target_length = target_end - target_start
    sig_size = x_clean.shape[-1]
    if np.random.rand() < 0.5:
        source_start = np.random.randint(0, target_start - target_length + 1)
    else:
        source_start = np.random.randint(target_end, sig_size - target_length + 1)

    source_end = source_start + target_length
    x_c, y_c, z_c = x_clean.copy(), y_clean.copy(), z_clean.copy()
    x_n, y_n, z_n = x_noisy.copy(), y_noisy.copy(), z_noisy.copy()

    x_tmp_c = x_c[target_start:target_end].copy()
    y_tmp_c = y_c[target_start:target_end].copy()
    z_tmp_c = z_c[target_start:target_end].copy()

    x_tmp_n = x_n[target_start:target_end].copy()
    y_tmp_n = y_n[target_start:target_end].copy()
    z_tmp_n = z_n[target_start:target_end].copy()

    x_c[target_start:target_end] = x_c[source_start:source_end]
    y_c[target_start:target_end] = y_c[source_start:source_end]
    z_c[target_start:target_end] = z_c[source_start:source_end]

    x_n[target_start:target_end] = x_n[source_start:source_end]
    y_n[target_start:target_end] = y_n[source_start:source_end]
    z_n[target_start:target_end] = z_n[source_start:source_end]

    x_c[source_start:source_end] = x_tmp_c
    y_c[source_start:source_end] = y_tmp_c
    z_c[source_start:source_end] = z_tmp_c

    x_n[source_start:source_end] = x_tmp_n
    y_n[source_start:source_end] = y_tmp_n
    z_n[source_start:source_end] = z_tmp_n

    return x_c, x_n, y_c, y_n, z_c, z_n


def extract_dc2_data(
    file_quads: List[Tuple[Path, Path, Path, Path]],
    mpe: float,
    min_zen: float,
    max_zen: float,
    swap_traces: bool,
    target_start: int,
    target_end: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Loop over file quadruplets, apply energy/zenith filters,
    optionally slice-swap, and accumulate x,y,z clean and noisy lists.
    """
    x_NJ, x_AN, y_NJ, y_AN, z_NJ, z_AN = ([] for _ in range(6))
    for adc_NJ, sd_NJ, adc_AN, sd_AN in tqdm(file_quads, desc="Processing files"):
        tadc_NJ = rt.DataFile(str(adc_NJ)).tadc_l1
        tsh_NJ  = rt.DataFile(str(sd_NJ)).tshower_l0
        tadc_AN = rt.DataFile(str(adc_AN)).tadc_l1
        tsh_AN  = rt.DataFile(str(sd_AN)).tshower_l0

        events = tadc_NJ.get_list_of_events()
        nb_events = len(events)
        event_counter = 0
        max_events_to_store = nb_events
        for evt, run in events:
            assert isinstance(evt, int)
            assert isinstance(run, int)  
            if event_counter < max_events_to_store:
                tsh_NJ.get_event(evt, run)
                tsh_AN.get_event(evt, run)

                zenith = tsh_NJ.zenith
                energy_primary = tsh_NJ.energy_primary

                event_counter += 1
                if energy_primary > mpe:
                    if min_zen <= zenith <= max_zen:
                        tadc_NJ.get_event(evt, run)
                        tadc_AN.get_event(evt, run)

                        traces_NJ = np.asarray(tadc_NJ.trace_ch, dtype=np.float32)
                        traces_AN = np.asarray(tadc_AN.trace_ch, dtype=np.float32)
                        for du in range(traces_NJ.shape[0]):
                            x0 = traces_NJ[du, 0]
                            y0 = traces_NJ[du, 1]
                            z0 = traces_NJ[du, 2]
                            
                            x1 = traces_AN[du, 0]
                            y1 = traces_AN[du, 1]
                            z1 = traces_AN[du, 2]
                            if (np.max(x0) / np.std(x1)) > 1 and (np.max(y0) / np.std(y1)) > 1 and (np.max(z0) / np.std(z1)) > 1:
                                if swap_traces:
                                    xc, xn, yc, yn, zc, zn = maybe_slice_swap_traces(
                                        x0, y0, z0, x1, y1, z1,
                                        target_start=target_start, target_end=target_end,
                                    )
                                    x_NJ.append(xc)
                                    x_AN.append(xn)
                                    y_NJ.append(yc)
                                    y_AN.append(yn)
                                    z_NJ.append(zc)
                                    z_AN.append(zn)
                                else: 
                                    x_NJ.append(x0)                                                                      
                                    x_AN.append(x1)
                                    y_NJ.append(y0)
                                    y_AN.append(y1)
                                    z_NJ.append(z0)
                                    z_AN.append(z1)
    return  x_AN, x_NJ, y_AN, y_NJ, z_AN, z_NJ


def save_split(
    data: Tuple[List[np.ndarray], ...],
    split: str,
    outdir: Path,
):
    """
    Stack and save noised and clean signals for a given split.
    """
    x_n, x_c, y_n, y_c, z_n, z_c = data
    noised = np.stack((x_n, y_n, z_n))
    clean  = np.stack((x_c, y_c, z_c))
    np.savez_compressed(outdir / f"dc2_{split}_noised_signals.npz", signals=noised)
    np.savez_compressed(outdir / f"dc2_{split}_clean_signals.npz",  signals=clean)
    logger.info(f"Saved {split} signals to {outdir}")


def parse_args():
    p = argparse.ArgumentParser(description="Process DC2 traces and save NPZ files.")
    p.add_argument("--save_folder", type=Path, required=True, help="Directory to save outputs")
    p.add_argument("--mpe", type=float, default=1e8, help="Minimum primary energy cut")
    p.add_argument("--min_zenith", type=float, default=0.0, help="Minimum zenith angle")
    p.add_argument("--max_zenith", type=float, default=89.0, help="Maximum zenith angle")
    p.add_argument("--target_start", type=int, default=240)
    p.add_argument("--target_end", type=int, default=400)
    return p.parse_args()


def main():
    args = parse_args()
    args.save_folder.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        split_dirs = DATA_CONFIG[split]
        swap_q, ns_q = get_matching_files_for_split(split_dirs, split)
        data_swap = extract_dc2_data(
            swap_q, args.mpe, args.min_zenith, args.max_zenith,
            swap_traces=True,
            target_start=args.target_start,
            target_end=args.target_end,
        )
        data_ns = extract_dc2_data(
            ns_q, args.mpe, args.min_zenith, args.max_zenith,
            swap_traces=False,
            target_start=args.target_start,
            target_end=args.target_end,
        )
        # concatenate swap and non-swap
        data = tuple(list(ds) + list(dns) for ds, dns in zip(data_swap, data_ns))
        save_split(data, split, args.save_folder)

if __name__ == "__main__":
    main()
