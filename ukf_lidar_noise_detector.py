import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from concurrent.futures import ProcessPoolExecutor
import laspy
from typing import List, Tuple, Optional, Dict

# UNSCENTED KALMAN FILTERING FOR GROUND CLEANING
# Update state dimension to 3 for [z, dz/dt, d2z/dt2]
STATE_DIM = 3
SIGMA_ALPHA = 0.01
SIGMA_BETA = 2.0
SIGMA_KAPPA = 0

GLOBAL_POINTS: Optional[MerweScaledSigmaPoints] = None

def init_process_points(n=STATE_DIM, alpha=SIGMA_ALPHA, beta=SIGMA_BETA, kappa=SIGMA_KAPPA):
    global GLOBAL_POINTS
    GLOBAL_POINTS = MerweScaledSigmaPoints(n, alpha, beta, kappa)

def fx_constant_acceleration(x: np.ndarray, dt: float) -> np.ndarray:
    F = np.array([
        [1, dt, 0.5 * dt**2],
        [0, 1, dt],
        [0, 0, 1]
    ])
    return F @ x

def hx_position_only(x: np.ndarray) -> np.ndarray:
    return np.array([x[0]])

def process_chunk(chunk: np.ndarray,
                  process_noise_var: float,
                  measurement_noise_var: float,
                  dt=1.0,
                  n_init=10) -> List[float]:
    if not len(chunk):
        return []

    if GLOBAL_POINTS is None:
        init_process_points()

    n_init = min(n_init, len(chunk))
    init_z = np.mean(chunk[:n_init])
    init_var = max(np.var(chunk[:n_init]), 1e-3)

    ukf = UnscentedKalmanFilter(
        dim_x=STATE_DIM,
        dim_z=1,
        dt=dt,
        fx=fx_constant_acceleration,
        hx=hx_position_only,
        points=GLOBAL_POINTS
    )

    # Initialize state: assume initial velocity and acceleration are zero
    ukf.x = np.array([init_z, 0.0, 0.0])
    ukf.P = np.diag([init_var, 1.0, 1.0])
    ukf.R = np.array([[measurement_noise_var]])
    ukf.Q = np.diag([process_noise_var]*STATE_DIM)

    zs = [np.array([z]) for z in chunk]

    try:
        xs, _ = ukf.batch_filter(zs)
        return [float(x[0]) for x in xs]
    except Exception as e:
        print("Exception in batch_filter:", e)
        return chunk.tolist()

def unscented_kalman_parallel(z_vals: np.ndarray,
                              process_noise_var: float,
                              measurement_noise_var: float,
                              chunk_size=500000,
                              dt=1.0,
                              n_init=10) -> np.ndarray:
    chunks = [z_vals[i:i + chunk_size] for i in range(0, len(z_vals), chunk_size)]
    results = [None] * len(chunks)
    with ProcessPoolExecutor(initializer=init_process_points) as executor:
        futures = [(i, executor.submit(process_chunk, c, process_noise_var, measurement_noise_var, dt, n_init))
                   for i, c in enumerate(chunks)]
        for i, f in futures:
            results[i] = f.result()
    return np.concatenate(results)

def apply_kalman_filtering_ground(las_data: laspy.LasData,
                                  ground_classes: List[int] = [2],
                                  noise_class_id: int = 18,
                                  chunk_size=500000,
                                  fix_thr=0.3,
                                  fix_q=0.01,
                                  fix_r=0.1,
                                  dt=1.0,
                                  n_init=10) -> Tuple[np.ndarray, float, Dict, np.ndarray, np.ndarray, np.ndarray]:
    z = las_data.z
    cls = las_data.classification
    gm = np.isin(cls, ground_classes)
    gz = z[gm]
    gi = np.where(gm)[0]

    sz = unscented_kalman_parallel(gz, fix_q, fix_r, chunk_size, dt, n_init)

    diff = np.abs(gz - sz)
    noise_mask = diff > fix_thr
    cls_upd = cls.copy()
    cls_upd[gi[noise_mask]] = noise_class_id
    noise_pct = np.mean(noise_mask) * 100

    params = {
        "threshold": fix_thr,
        "process_noise_Q": fix_q,
        "measurement_noise_R": fix_r,
        "dynamic_params_used": False
    }
    return cls_upd, noise_pct, params, gz, sz, gi

def save_classified_las(las_data: laspy.LasData, classification: np.ndarray, output_file: str):
    new_las = laspy.LasData(header=las_data.header)
    new_las.points = las_data.points.copy()
    new_las.classification = classification
    new_las.write(output_file)

def main():
    parser = argparse.ArgumentParser(
        description="UKF-based ground noise detection in LAS/LAZ files"
    )
    parser.add_argument(
        "input_file",
        help="Path to input LAS/LAZ file"
    )
    parser.add_argument(
        "output_file",
        help="Path to write the reclassified LAS file"
    )
    parser.add_argument(
        "--ground_classes", "-g",
        type=int,
        nargs="+",
        default=[2],
        help="List of classes to treat as ground (default: 2)"
    )
    parser.add_argument(
        "--noise_class_id", "-n",
        type=int,
        default=19,
        help="Classification ID to assign to detected noise (default: 19)"
    )
    parser.add_argument(
        "--chunk_size", "-c",
        type=int,
        default=250_000,
        help="Number of points per UKF chunk (default: 250000)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.075,
        help="Elevation deviation threshold (m) (default: 0.075)"
    )
    parser.add_argument(
        "--process_noise_Q", "-q",
        type=float,
        default=8.0,
        help="Process‐noise variance Q (default: 8.0)"
    )
    parser.add_argument(
        "--measurement_noise_R", "-r",
        type=float,
        default=0.20,
        help="Measurement‐noise variance R (default: 0.20)"
    )
    parser.add_argument(
        "--dt", "-d",
        type=float,
        default=1.0,
        help="Time step Δt (default: 1.0)"
    )
    parser.add_argument(
        "--n_init", "-i",
        type=int,
        default=10,
        help="Number of initial points to estimate state (default: 10)"
    )
    args = parser.parse_args()

    start = time.time()
    las = laspy.read(args.input_file)
    # sort by gps_time
    idx = np.argsort(las.gps_time)
    for attr in ("x","y","z","classification","gps_time"):
        setattr(las, attr, getattr(las, attr)[idx])

    updated_cls, noise_pct, params, gz, sz, gi = apply_kalman_filtering_ground(
        las_data=las,
        ground_classes=args.ground_classes,
        noise_class_id=args.noise_class_id,
        chunk_size=args.chunk_size,
        fix_thr=args.threshold,
        fix_q=args.process_noise_Q,
        fix_r=args.measurement_noise_R,
        dt=args.dt,
        n_init=args.n_init
    )

    print(f"Detected noise: {noise_pct:.2f}%")
    save_classified_las(las, updated_cls, args.output_file)
    print(f"Saved classified LAS to {args.output_file}")
    print(f"Params used: {params}")
    print(f"Elapsed time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()
