import argparse
import glob
import os

import laspy
import numpy as np

def kalman_filter_1d_chunk(noisy_signal, process_var, measurement_var):
    n = len(noisy_signal)
    smoothed_signal = np.zeros_like(noisy_signal)
    P = 1.0
    Q = process_var
    R = measurement_var
    x = noisy_signal[0]
    for i in range(n):
        x = x  # Prediction step
        P = P + Q
        K = P / (P + R)  #Kalman gain
        x = x + K * (noisy_signal[i] - x)  # Update estimate
        P = (1 - K) * P
        smoothed_signal[i] = x
    return smoothed_signal

def apply_kalman_filtering(z_values,
                           classification,
                           ground_classes=(2, 7),
                           noise_threshold=13.0,
                           process_var=1e-1,
                           measurement_var=2.0,
                           class_id=18):
    """
    Applies Kalman filtering only to points whose original class is in ground_classes,
    and marks as noise (class_id) any of those whose residual exceeds noise_threshold.

    Parameters:
        z_values (array): Array of z values from LiDAR points.
        classification (array): Array of classification IDs.
        ground_classes (tuple): Only points in these classes are checked for noise.
        noise_threshold (float): Threshold for |z - smoothed_z| to flag noise.
        process_var (float): Process variance for the Kalman filter.
        measurement_var (float): Measurement variance for the Kalman filter.
        class_id (int): Classification ID to assign to detected noise.

    Returns:
        classification (array): Updated classification array.
        noise_percentage (float): Percentage of ground-class points marked as noise.
    """
    z_values = np.asarray(z_values)
    classification = np.asarray(classification)

    # Mask for only those points originally in ground_classes
    ground_mask = np.isin(classification, ground_classes)
    if not ground_mask.any():
        print("No ground-class points to filter.")
        return classification, 0.0

    # Extract just the ground elevations to filter
    z_ground = z_values[ground_mask]
    smoothed = kalman_filter_1d_chunk(z_ground, process_var, measurement_var).astype(np.float32)

    # Compute residuals and flag noise peaks
    residuals = np.abs(z_ground - smoothed)
    noise_peaks = residuals > noise_threshold

    # Map back into the full classification array
    ground_indices = np.nonzero(ground_mask)[0]
    classification[ground_indices[noise_peaks]] = class_id

    noise_pct = 100.0 * noise_peaks.sum() / ground_mask.sum()
    print(f"{noise_pct:.2f}% of ground-class points flagged as noise.")

    return classification, noise_pct


def save_classified_las(las_data, classification, output_file):
    """
    Saves LiDAR data into a new LAS/LAZ file with updated classifications.
    """
    print(f"Saving modified LiDAR data to {output_file}...")
    try:
        las_data.classification = classification
        las_data.z = las_data.z.copy()
        las_data.write(output_file)
        print(f"Reclassified LAS file successfully saved as {output_file}.")
    except Exception as e:
        print(f"Failed to save reclassified LAS data: {e}")

def process_file_kalman(file_path, output_dir,
                        ground_classes=(2),
                        noise_threshold=13.0,
                        process_var=1e-1,
                        measurement_var=2.0,
                        class_id=18):
    """
    Read a LAS/LAZ, sort all points by gps_time, then apply Kalman-based noise detection
    on class 2 only, finally write a new file with exactly the same point order.
    """
    print(f"Processing file: {file_path}")
    try:
        las = laspy.read(file_path)
    except Exception as e:
        print(f"  ✖ Failed to read {file_path}: {e}")
        return None

    # sort the entire point table by gps_time, so all dims stay aligned
    order = np.argsort(las.gps_time)
    las.points = las.points[order]

    # apply Kalman filtering
    cls, noise_pct = apply_kalman_filtering(
        z_values=las.z,
        classification=las.classification,
        ground_classes=ground_classes,
        noise_threshold=noise_threshold,
        process_var=process_var,
        measurement_var=measurement_var,
        class_id=class_id
    )

    # write out to a new file
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"kalmin_{os.path.basename(file_path)}")
    las.classification = cls
    las.write(out_file)
    print(f"  ✔ {noise_pct:.2f}% of ground points flagged as noise; saved to {out_file}")

    return noise_pct


def process_folder(input_folder, output_dir,
                                  output_kalman_file,
                                  noise_threshold=13.0):
    """
    Analyze all LAS/LAZ files in a folder using both Kalman Filtering and EMD.
    Writes high-noise files detected by each method to separate text files.

    Parameters:
        input_folder (str): Path to folder containing LAS/LAZ files.
        output_kalman_file (str): Path to text file listing high-noise files by Kalman Filtering.
    """
    las_files = glob.glob(os.path.join(input_folder, "*.las")) + \
                glob.glob(os.path.join(input_folder, "*.laz"))

    if not las_files:
        return

    percents = []

    for file_path in las_files:
        file_name = os.path.basename(file_path)

        # Process Kalman Filtering
        noise_kalman = process_file_kalman(file_path, output_dir)
        if noise_kalman is not None:
            percents.append((file_name, noise_kalman))

    # Write results to Kalman txt file
    try:
        with open(output_kalman_file, 'w') as f_kalman:
            for file_name, noise_pct in percents:
                f_kalman.write(f"{file_name}: {noise_pct:.2f}% noise\n")
    except Exception as e:
        return


def main():
    parser = argparse.ArgumentParser(
        description="Batch Kalman-based noise detection on LAS/LAZ files"
    )
    parser.add_argument(
        "input_folder",
        help="Path to folder containing .las/.laz files to process"
    )
    parser.add_argument(
        "output_dir",
        help="Directory where reclassified LAS/LAZ files will be saved"
    )
    parser.add_argument(
        "output_kalman_file",
        help="Path to text file to write per-file noise percentages"
    )
    parser.add_argument(
        "--noise_threshold", "-t",
        type=float,
        default=13.0,
        help="|z–ẑ| threshold (m) above which points are flagged as noise (default: 13.0)"
    )
    args = parser.parse_args()

    process_folder(
        input_folder=args.input_folder,
        output_dir=args.output_dir,
        output_kalman_file=args.output_kalman_file,
        noise_threshold=args.noise_threshold
    )

if __name__ == "__main__":
    main()
