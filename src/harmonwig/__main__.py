#!/usr/bin/env python
"""Wigner sampling of harmonic vibrational wavefunction.

Authors:
    * Daniel Hollas <daniel.hollas@bristol.ac.uk>
"""

from .harmonwig import HarmonicWigner


def parse_cmd():
    import argparse

    desc = "Program for harmonic Wigner sampling"
    prog = "harmonwig"
    parser = argparse.ArgumentParser(description=desc, prog=prog)
    parser.add_argument(
        "input_file", metavar="INPUT_FILE", help="Output file from ab initio program."
    )
    parser.add_argument(
        "-n",
        "--nsamples",
        type=int,
        default=1,
        help="Number of Wigner samples",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=42424242,
        help="Random seed",
    )
    parser.add_argument(
        "--freqthr",
        dest="low_freq_thr",
        default=0.0,
        type=float,
        help="Low-frequency threshold",
    )

    """
    parser.add_argument(
        "--file-format",
        dest="file_fmt",
        default="auto",
        help="Format of the output file",
    )
    """

    parser.add_argument(
        "-o",
        "--output-file",
        dest="output_fname",
        default="harmonic_samples.xyz",
        help="Output file name",
    )

    return parser.parse_args()


def error(msg: str):
    import sys

    print(f"ERROR: {msg}")
    sys.exit(1)


def read_qm_output(fname: str) -> dict:
    from pathlib import Path

    from cclib.io import ccread

    # TODO: Determine the QM program and have an allowlist
    # of known-to-work programgs
    path = Path(fname)
    try:
        with path.open("r") as f:
            parsed_obj = ccread(f)
    except FileNotFoundError as e:
        error(str(e))

    if parsed_obj is None:
        error("Could not read QM output file")
    return validate(parsed_obj)


def validate(parsed_obj) -> dict:
    d = parsed_obj.getattributes()
    if not d["optdone"]:
        error("Geometry optimization did not finish!")
    req_keys = ["atomcoords", "atommasses", "vibdisps", "vibfreqs"]
    for key in req_keys:
        if key not in d:
            msg = f"Incomplete frequency data in the output file, missing key '{key}'"
            error(msg)
    # TODO: We should return a custom dataclass instead of full cclib dictionary
    return d


def main():
    """Entry point to the CLI app"""

    opts = parse_cmd()

    out = read_qm_output(opts.input_file)

    import ase
    from tqdm import tqdm

    # We assume that the last coordinates are the optimized ones
    coords = out["atomcoords"][0]
    ase_mol = ase.Atoms(
        numbers=out["atomnos"], positions=coords, masses=out["atommasses"], pbc=False
    )

    print("Normal mode frequencies [cm^-1]:")
    print(out["vibfreqs"])
    print("Atom masses:")
    print(out["atommasses"])

    wigner = HarmonicWigner(
        ase_mol,
        out["vibfreqs"],
        out["vibdisps"],
        seed=opts.seed,
        low_freq_thr=opts.low_freq_thr,
    )

    print(f"Generating {opts.nsamples} samples to {opts.output_fname}")
    barfmt = "{l_bar}{bar}|{n_fmt}/{total_fmt}    "
    wigner_samples = []
    for _ in tqdm(range(opts.nsamples), delay=0.3, colour="green", bar_format=barfmt):
        wigner_samples.append(wigner.get_ase_sample())

    ase.io.write(
        opts.output_fname,
        images=wigner_samples,
        format="extxyz",
    )
