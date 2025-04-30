import argparse

import numpy as np
import laspy
from scipy.spatial import cKDTree
from collections import deque
import numba
from numba import jit, prange
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@jit(nopython=True, parallel=True)
def compute_grid_indices_numba(points, grid_size, mins):
    """Compute 3D grid cell indices with Numba acceleration"""
    indices = np.empty((points.shape[0], 3), dtype=np.int32)
    for i in prange(points.shape[0]):
        indices[i] = np.floor((points[i] - mins) / grid_size).astype(np.int32)
    return indices

@jit(nopython=True)
def compute_projections(points, current_point, unit_direction):
    """Compute projections of vectors onto unit direction with Numba"""
    projections = np.empty(len(points), dtype=np.float32)
    for i in range(len(points)):
        vec = points[i] - current_point
        proj = np.dot(vec, unit_direction)
        projections[i] = proj if proj > 0 else -np.inf
    return projections

class CloudFinder:
    def __init__(self, points, grid_size=20.0, distance_threshold=5.0):
        """
        Initialize CloudFinder with LiDAR points and parameters.
        :param points: Nx3 numpy array containing LiDAR data.
        :param grid_size: Edge length (l) of the 3D grid cells (meters).
        :param distance_threshold: Distance threshold (d) for connectivity (meters).
        """
        self.points = points.astype(np.float32)
        self.grid_size = grid_size
        self.distance_threshold = distance_threshold
        self.visited = np.zeros(len(points), dtype=bool)
        self.regions = -1 * np.ones(len(points), dtype=np.int32)
        self.lock = threading.Lock()  # For safe concurrent updates

        # Build KD-tree with leaf size optimization for larger datasets
        leaf_size = min(40, max(10, int(len(points) / 10000)))
        t_start = time.time()
        self.tree = cKDTree(self.points, leafsize=leaf_size)
        logging.info(f"KD-tree built in {time.time() - t_start:.2f} seconds")

        self.region_id = 0

        # Pre-compute unit direction vectors for directional shortcuts
        self.unit_direction_vectors = self.get_unit_direction_vectors()

        # Cache for grid indices
        self._grid_indices = None
        self._grid_mins = None

    def compute_grid_indices(self):
        """
        Compute 3D grid cell indices for each point using floor division.
        Uses cached values if already computed.
        Returns:
            indices: Array of grid cell indices for each point.
            mins: Minimum coordinates for each dimension.
        """
        if self._grid_indices is not None and self._grid_mins is not None:
            return self._grid_indices, self._grid_mins

        mins = np.min(self.points, axis=0)
        if len(self.points) > 10000:
            indices = compute_grid_indices_numba(self.points, self.grid_size, mins)
        else:
            indices = np.floor((self.points - mins) / self.grid_size).astype(np.int32)

        self._grid_indices = indices
        self._grid_mins = mins
        return indices, mins

    def select_initial_seeds(self, n_seeds=5):
        """
        Select n_seeds initial seed points using histogram and grid cell density analysis.
        Returns:
            seeds: List of tuples (seed_index, seed_point) for the initial seeds.
        """
        t_start = time.time()
        indices, mins = self.compute_grid_indices()

        # Use only Z values for histogram analysis (faster)
        z_vals = self.points[:, 2]
        bin_count = min(10, max(3, int(len(self.points) / 100000)))
        hist, bin_edges = np.histogram(z_vals, bins=bin_count)
        max_bin_index = np.argmax(hist)
        z_lower = bin_edges[max_bin_index]
        z_upper = bin_edges[max_bin_index + 1]

        candidate_mask = (z_vals >= z_lower) & (z_vals < z_upper)
        if np.sum(candidate_mask) > 100000:
            sample_rate = max(0.1, 100000 / np.sum(candidate_mask))
            sample_mask = np.random.random(len(candidate_mask)) < sample_rate
            candidate_mask = candidate_mask & sample_mask

        candidate_indices = indices[candidate_mask]
        if candidate_indices.size == 0:
            candidate_indices = indices[::max(1, len(indices) // 10000)]

        # Create a unique representation of each grid cell
        dtype = np.dtype((np.void, candidate_indices.dtype.itemsize * candidate_indices.shape[1]))
        candidate_view = candidate_indices.view(dtype).reshape(-1)
        unique_cells, unique_idx, counts = np.unique(candidate_view, return_index=True, return_counts=True)
        sorted_order = np.argsort(-counts)

        seeds = []
        for i in range(min(n_seeds, len(sorted_order))):
            idx = unique_idx[sorted_order[i]]
            grid_cell = candidate_indices[idx]
            cell_center = grid_cell * self.grid_size + (self.grid_size / 2) + mins
            distance, seed_idx = self.tree.query(cell_center)
            seed_point = self.points[seed_idx]
            seeds.append((seed_idx, seed_point))
        logging.info(f"Initial seed selection completed in {time.time() - t_start:.2f} seconds")
        return seeds

    def get_unit_direction_vectors(self):
        """
        Define the 14 main directional unit vectors (6 primary and 8 auxiliary).
        Returns:
            List of 14 numpy arrays representing the unit directional vectors.
        """
        d = self.distance_threshold
        primary = [
            np.array([d, 0, 0], dtype=np.float32),
            np.array([-d, 0, 0], dtype=np.float32),
            np.array([0, d, 0], dtype=np.float32),
            np.array([0, -d, 0], dtype=np.float32),
            np.array([0, 0, d], dtype=np.float32),
            np.array([0, 0, -d], dtype=np.float32)
        ]
        factor = d * (np.sqrt(2) / 2)
        aux = [
            np.array([factor, factor, 0], dtype=np.float32),
            np.array([-factor, factor, 0], dtype=np.float32),
            np.array([factor, -factor, 0], dtype=np.float32),
            np.array([-factor, -factor, 0], dtype=np.float32),
            np.array([factor, 0, factor], dtype=np.float32),
            np.array([-factor, 0, factor], dtype=np.float32),
            np.array([factor, 0, -factor], dtype=np.float32),
            np.array([-factor, 0, -factor], dtype=np.float32)
        ]
        directions = primary + aux
        unit_vectors = [vec / np.linalg.norm(vec) for vec in directions]
        return unit_vectors

    def region_growing(self, initial_index):
        """
        Perform region growing starting from a single seed.
        Updates the global visited and regions arrays using locks.
        """
        t_start = time.time()
        queue = deque([initial_index])
        with self.lock:
            self.visited[initial_index] = True
            self.regions[initial_index] = self.region_id

        points_processed = 1
        total_points = len(self.points)
        last_report_time = time.time()
        report_interval = 5.0  # seconds

        while queue:
            batch_size = min(1000, len(queue))
            current_batch = [queue.popleft() for _ in range(batch_size)]
            for current_index in current_batch:
                current_point = self.points[current_index]
                neighbor_indices = self.tree.query_ball_point(current_point, self.distance_threshold)
                if neighbor_indices:
                    neighbor_indices = np.array(neighbor_indices, dtype=np.int32)
                    # Update unvisited neighbors in a thread-safe way
                    with self.lock:
                        unvisited_mask = ~self.visited[neighbor_indices]
                        new_neighbors = neighbor_indices[unvisited_mask]
                        if new_neighbors.size > 0:
                            self.visited[new_neighbors] = True
                            self.regions[new_neighbors] = self.region_id
                    queue.extend(new_neighbors.tolist())
                    points_processed += len(new_neighbors)

                    # Process directional vectors in batch for significant neighborhoods
                    if len(neighbor_indices) > 10:
                        neighbor_points = self.points[neighbor_indices]
                        for unit_direction in self.unit_direction_vectors:
                            if len(neighbor_indices) > 100:
                                projections = compute_projections(neighbor_points, current_point, unit_direction)
                            else:
                                vecs = neighbor_points - current_point
                                projections = np.dot(vecs, unit_direction)
                                projections[projections <= 0] = -np.inf
                            if np.max(projections) > 0:
                                max_idx = np.argmax(projections)
                                farthest_index = neighbor_indices[max_idx]
                                with self.lock:
                                    if not self.visited[farthest_index]:
                                        self.visited[farthest_index] = True
                                        self.regions[farthest_index] = self.region_id
                                        queue.append(farthest_index)
                                        points_processed += 1

            current_time = time.time()
            if current_time - last_report_time > report_interval:
                progress = points_processed / total_points * 100
                elapsed = current_time - t_start
                logging.info(f"[Region {self.region_id}] {progress:.1f}% complete "
                             f"({points_processed:,}/{total_points:,} points), elapsed: {elapsed:.1f}s, "
                             f"queue size: {len(queue):,}")
                last_report_time = current_time

        logging.info(f"Region {self.region_id} growing completed in {time.time() - t_start:.2f} seconds, "
                     f"{points_processed:,} points processed")
        return

    def detect_noise(self):
        """
        Identify points that were not reached by any region.
        Returns:
            noise_indices: Array of indices corresponding to noise points.
        """
        noise_indices = np.where(~self.visited)[0]
        return noise_indices

def process_las_file(input_path, output_path, grid_size=20.0, distance_threshold=5.0):
    """Process a LAS file with performance metrics using parallel region growing."""
    t_total_start = time.time()

    logging.info(f"Processing file: {input_path}")
    logging.info(f"Parameters: grid_size={grid_size}, distance_threshold={distance_threshold}")

    t_read_start = time.time()
    try:
        las = laspy.read(input_path)
        read_time = time.time() - t_read_start
        logging.info(f"Read {len(las.points):,} points in {read_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Error reading LAS file: {e}")
        return

    t_extract_start = time.time()
    points = np.vstack((las.x, las.y, las.z)).transpose()
    extract_time = time.time() - t_extract_start
    logging.info(f"Extracted point data in {extract_time:.2f} seconds")

    cf = CloudFinder(points, grid_size=grid_size, distance_threshold=distance_threshold)
    seeds = cf.select_initial_seeds(n_seeds=5)
    logging.info(f"Selected {len(seeds)} initial seed points.")

    # Parallel region growing across seeds using ThreadPoolExecutor.
    # We assume that regions grown from these seeds are largely independent.
    futures = []
    with ThreadPoolExecutor(max_workers=len(seeds)) as executor:
        for seed_index, seed_point in seeds:
            with cf.lock:
                if not cf.visited[seed_index]:
                    cf.region_id += 1  # assign a new region id
                    current_region = cf.region_id
                    logging.info(f"Starting region growing for seed point {seed_point} as Region {current_region}")
                    # Submit the region growing task
                    futures.append(executor.submit(cf.region_growing, seed_index))
        # Wait for all region growing tasks to finish
        for future in futures:
            future.result()

    t_noise_start = time.time()
    noise_indices = cf.detect_noise()
    noise_time = time.time() - t_noise_start
    logging.info(f"Detected {len(noise_indices):,} noise points in {noise_time:.2f} seconds")

    t_write_start = time.time()
    las.classification[noise_indices] = 19  # mark noise points with classification 19
    las.write(output_path)
    write_time = time.time() - t_write_start
    logging.info(f"Updated and wrote output in {write_time:.2f} seconds")

    total_time = time.time() - t_total_start
    logging.info(f"Total processing time: {total_time:.2f} seconds")
    logging.info(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="CloudFinder: detect noise in a LAS/LAZ file via region‐growing"
    )
    parser.add_argument(
        "input_file",
        help="Path to input LAS/LAZ file"
    )
    parser.add_argument(
        "output_file",
        help="Path to write the classified output file"
    )
    parser.add_argument(
        "--grid_size", "-g",
        type=float,
        default=20.0,
        help="Edge length (m) of the 3D grid cells (default: 20.0)"
    )
    parser.add_argument(
        "--distance_threshold", "-d",
        type=float,
        default=5.0,
        help="Connectivity threshold (m) for region growing (default: 5.0)"
    )
    args = parser.parse_args()

    process_las_file(
        args.input_file,
        args.output_file,
        grid_size=args.grid_size,
        distance_threshold=args.distance_threshold
    )

if __name__ == "__main__":
    main()
