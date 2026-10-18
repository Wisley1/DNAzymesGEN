#!/usr/bin/env python3
"""Reproduce Data/Negatives/dna_sequences.csv from the methods description.

Lengths are sampled from a Beta distribution with alpha = 7.0 and beta = 7.5
on [0, 1], linearly rescaled to [8, 74], turned into integers by stochastic
rounding (add 1 with probability equal to the fractional part), and clipped
to [8, 74]. Each sequence is then drawn i.i.d. and uniformly from A, C, G, T.

All draws use NumPy's default Generator with SEED = 12345. Every length is
drawn before any nucleotide, which is the stream that matches the released
CSV. The writer uses CRLF so the bytes match dna_sequences.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SEED = 12345
N_SEQUENCES = 250_000
LENGTH_MIN = 8
LENGTH_MAX = 74
ALPHA = 7.0
BETA = 7.5
ALPHABET = np.array(list("ACGT"))


def generate_negative_sequences(
    n: int = N_SEQUENCES,
    seed: int = SEED,
) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    unit = rng.beta(ALPHA, BETA, size=n)
    continuous = LENGTH_MIN + unit * (LENGTH_MAX - LENGTH_MIN)
    integer_part = np.floor(continuous)
    fractional = continuous - integer_part
    lengths = integer_part + (rng.random(n) < fractional)
    lengths = np.clip(lengths, LENGTH_MIN, LENGTH_MAX).astype(np.int64)
    sequences = [
        "".join(rng.choice(ALPHABET, size=int(length))) for length in lengths
    ]
    return lengths, sequences


def write_csv(path: Path, lengths: np.ndarray, sequences: list[str]) -> None:
    lines = ["id,length,sequence"]
    lines.extend(
        f"{i},{int(length)},{sequence}"
        for i, (length, sequence) in enumerate(zip(lengths, sequences), start=1)
    )
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("ascii"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("dna_sequences.csv"),
        help="Destination CSV (default: ./dna_sequences.csv)",
    )
    parser.add_argument("--n", type=int, default=N_SEQUENCES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    lengths, sequences = generate_negative_sequences(n=args.n, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, lengths, sequences)


if __name__ == "__main__":
    main()
