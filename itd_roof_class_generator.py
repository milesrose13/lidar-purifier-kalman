#!/usr/bin/env python3
import laspy
import numpy as np
import time
from ITD import itd

def save_classified_las(las_data, classification, output_file):
    """
    Saves LiDAR data into a new LAS file with updated classifications.

    Args:
        las_data (laspy.LasData): The original LAS data.
        classification (ndarray): The classification labels for each point.
        output_file (str): Path to save the output LAS file.
    """
    print(f"Saving modified LiDAR data to {output_file}...")
    try:
        las_data.classification = classification
        las_data.write(output_file)
    except Exception as e:
        raise RuntimeError(f"Failed to save modified LAS data: {e}")
    print(f"Modified LAS file successfully saved as {output_file}.")


def main(input_file, output_file, percentile=5.0, height_offset=2.0):
    """
    Classify roofs by applying ITD to single-return points,
    thresholding low IMF1 values and requiring a minimum height above ground.
    """
    start_time = time.time()

    # Load LAS (assumes ground points already classed as 2)
    las = laspy.read(input_file)
    z = las.z
    existing_cls = las.classification.copy()

    # Identify single-return points only
    sr_mask = (las.return_number == 1) & (las.num_returns == 1)
    count_sr = int(np.count_nonzero(sr_mask))
    if count_sr == 0:
        raise RuntimeError("No single-return points found.")
    idx_sr = np.nonzero(sr_mask)[0]
    z_sr = z[sr_mask]
    print(f"Found {count_sr} single-return points.")

    # Run ITD on single-return z-values
    print("Running ITD on single-return points...")
    IMF_sr = itd(np.asarray(z_sr))
    IMF1_sr = IMF_sr[0]

    # Threshold lowest IMF1 percentile
    thr = np.percentile(IMF1_sr, percentile)
    print(f"IMF1 threshold at {percentile}th percentile: {thr:.4f}")

    # Compute ground elevation baseline
    ground_z = z[existing_cls == 2]
    count_ground = int(ground_z.shape[0])
    if count_ground > 0:
        ground_elev = float(np.median(ground_z))
        print(f"Median ground elevation: {ground_elev:.2f}")
    else:
        ground_elev = float(np.min(z_sr))
        print("Warning: no ground class found, using min SR elevation as ground baseline.")

    # Select roof candidates: low IMF1 and height above ground
    roof_mask_sr = (IMF1_sr <= thr) & (z_sr > ground_elev + height_offset)
    roof_indices = idx_sr[roof_mask_sr]
    count_roof = int(roof_indices.shape[0])
    print(f"Identified {count_roof} roof candidates.")

    # Merge into final classification
    final_cls = existing_cls.copy()
    final_cls[roof_indices] = 6

    # Save output LAS
    save_classified_las(las, final_cls, output_file)

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.2f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Roof classification using ITD on single-return points with height filter"
    )
    parser.add_argument("input", help="input LAS/LAZ file with pre-classified ground")
    parser.add_argument("output", help="output LAS file with roof points labeled as 6")
    parser.add_argument(
        "--percentile", type=float, default=5.0,
        help="IMF1 percentile cutoff for roof detection (default: 5.0)"
    )
    parser.add_argument(
        "--height_offset", type=float, default=2.0,
        help="minimum height above ground (m) for roof detection (default: 2.0)"
    )
    args = parser.parse_args()
    main(args.input, args.output, percentile=args.percentile, height_offset=args.height_offset)
