#!/usr/bin/env python3
"""
Generate gt.ibin (PipeANN / IndexDiskANN ground truth format).

This script converts either:
  1. A DiskANN ground truth file (e.g. gt_100.bin, containing [n, k] header,
     uint32 IDs, and float32 distances; total size = 8 + 2 * n * k * 4 bytes).
  2. An INRIA Texmex ivecs file (e.g. sift100m_gtruth.ivecs, containing
     row-wise [int32 d, d * int32 IDs]).

Output format (gt.ibin):
  - 4 bytes: int32 n (number of queries)
  - 4 bytes: int32 k (number of neighbors, e.g. 100)
  - n * k * 4 bytes: contiguous uint32 neighbor IDs

Usage:
  ./gen_gt_ibin.py [INPUT] [OUTPUT]

Examples:
  ./gen_gt_ibin.py ~/Datasets/sift100m/diskann/gt_100.bin ~/pipeann_sift100m/gt.ibin
  ./gen_gt_ibin.py ~/pipeann_sift100m/sift100m_gtruth.ivecs ~/pipeann_sift100m/gt.ibin
  ./gen_gt_ibin.py  # auto-discovers input and writes to $DATA/gt.ibin
"""

import argparse
import glob
import os
import sys
import numpy as np


def find_default_input():
    home = os.path.expanduser("~")
    root = os.environ.get("ROOT", home)
    data = os.environ.get("DATA", os.path.join(root, "pipeann_sift100m"))

    candidates = [
        os.path.join(data, "gt_100.bin"),
        os.path.join(data, "gt_100.fbin"),
        os.path.join(data, "sift100m_gtruth.ivecs"),
        os.path.join(data, "bigann_gnd.ivecs"),
        os.path.join(home, "Datasets", "sift100m", "diskann", "gt_100.bin"),
        os.path.join(home, "Datasets", "sift100m", "diskann", "gt_100.fbin"),
    ]

    # Also search Hugging Face cache if present
    hf_glob = os.path.join(
        home,
        ".cache",
        "huggingface",
        "hub",
        "datasets--maknee--sift100m",
        "snapshots",
        "*",
        "diskann",
        "gt_100.bin",
    )
    candidates.extend(glob.glob(hf_glob))

    for path in candidates:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return os.path.realpath(path)
    return None


def find_default_output():
    root = os.environ.get("ROOT", os.path.expanduser("~"))
    data = os.environ.get("DATA", os.path.join(root, "pipeann_sift100m"))
    return os.path.join(data, "gt.ibin")


def convert_diskann_to_ibin(input_path, output_path):
    """Convert DiskANN gt_100.bin (IDs + distances) to gt.ibin (IDs only)."""
    file_size = os.path.getsize(input_path)
    if file_size < 8:
        raise ValueError(f"File {input_path} is too small to contain a valid header ({file_size} bytes)")

    hdr = np.fromfile(input_path, dtype=np.int32, count=2)
    n, k = int(hdr[0]), int(hdr[1])
    expected_da_size = 8 + 2 * n * k * 4  # header + IDs (uint32) + dists (float32)
    expected_ibin_size = 8 + n * k * 4    # header + IDs (uint32)

    if file_size == expected_da_size:
        print(f"[gen_gt_ibin] Input is DiskANN format: {n} queries, top-{k} neighbors with float distances.")
    elif file_size == expected_ibin_size:
        print(f"[gen_gt_ibin] Input is already in .ibin format: {n} queries, top-{k} neighbors.")
    else:
        print(f"[gen_gt_ibin] Warning: file size ({file_size}) does not match standard DiskANN ({expected_da_size}) "
              f"or ibin ({expected_ibin_size}). Attempting to read {n} x {k} IDs.")

    ids = np.fromfile(input_path, dtype=np.uint32, count=n * k, offset=8)
    if len(ids) != n * k:
        raise RuntimeError(f"Short read: expected {n * k} IDs, read {len(ids)}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(hdr.tobytes())
        f.write(ids.tobytes())
    os.replace(tmp_path, output_path)

    out_size = os.path.getsize(output_path)
    print(f"[gen_gt_ibin] Wrote {output_path} ({n} queries, k={k}, {out_size} bytes)")


def convert_ivecs_to_ibin(input_path, output_path):
    """Convert Texmex .ivecs format to gt.ibin."""
    file_size = os.path.getsize(input_path)
    if file_size < 8:
        raise ValueError(f"File {input_path} is too small ({file_size} bytes)")

    d = int(np.fromfile(input_path, dtype=np.int32, count=1)[0])
    row_bytes = (d + 1) * 4
    if file_size % row_bytes != 0:
        raise ValueError(f"File size {file_size} is not divisible by row size {row_bytes} (d={d})")

    n = file_size // row_bytes
    print(f"[gen_gt_ibin] Input is .ivecs format: {n} queries, dimension/k={d}.")

    mem = np.memmap(input_path, dtype=np.int32, mode="r").reshape(n, d + 1)
    ids = mem[:, 1:].astype(np.uint32)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    hdr = np.array([n, d], dtype=np.int32)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(hdr.tobytes())
        f.write(ids.tobytes())
    os.replace(tmp_path, output_path)

    out_size = os.path.getsize(output_path)
    print(f"[gen_gt_ibin] Wrote {output_path} ({n} queries, k={d}, {out_size} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Generate gt.ibin from DiskANN gt.bin or Texmex .ivecs")
    parser.add_argument("input", nargs="?", default=None, help="Input ground truth file (gt_100.bin, .ivecs, etc.)")
    parser.add_argument("output", nargs="?", default=None, help="Output gt.ibin destination file")
    args = parser.parse_args()

    input_path = args.input or find_default_input()
    if not input_path or not os.path.isfile(input_path):
        print("[gen_gt_ibin] Error: Could not locate a ground truth input file.", file=sys.stderr)
        print("Please specify an input file, e.g.:", file=sys.stderr)
        print("  ./gen_gt_ibin.py /path/to/gt_100.bin [/path/to/gt.ibin]", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or find_default_output()

    print(f"[gen_gt_ibin] Input:  {input_path}")
    print(f"[gen_gt_ibin] Output: {output_path}")

    # Determine input type
    if input_path.endswith(".ivecs"):
        convert_ivecs_to_ibin(input_path, output_path)
    else:
        convert_diskann_to_ibin(input_path, output_path)


if __name__ == "__main__":
    main()
