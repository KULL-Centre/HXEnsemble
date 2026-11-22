"""Utilities for HDX analysis."""

import numpy as np

def find_elbow(x, y):
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Sort x (and y) if needed, but keep track of original indices
    if not np.all(np.diff(x) >= 0):
        idx_sort = np.argsort(x)
        x_sorted = x[idx_sort]
        y_sorted = y[idx_sort]
    else:
        idx_sort = np.arange(len(x))
        x_sorted, y_sorted = x, y

    # Line between first and last point
    p1 = np.array([x_sorted[0], y_sorted[0]])
    p2 = np.array([x_sorted[-1], y_sorted[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    # Distances from each point to the line
    vec_from_p1 = np.vstack([x_sorted - p1[0], y_sorted - p1[1]]).T
    proj_len = np.dot(vec_from_p1, line_vec_norm)
    proj_point = np.outer(proj_len, line_vec_norm) + p1
    dist = np.linalg.norm(vec_from_p1 - (proj_point - p1), axis=1)

    # Elbow = point with maximum distance
    idx_elbow_sorted = np.argmax(dist)
    idx_elbow_original = idx_sort[idx_elbow_sorted]

    return idx_elbow_original, x[idx_elbow_original], y[idx_elbow_original]


def find_xtc_filenames(path_to_dir, suffix=".xtc"):
    """Names of trajectory files in a directory, sorted reproducibly."""
    from pathlib import Path
    return sorted(p.name for p in Path(path_to_dir).iterdir()
                  if p.is_file() and p.name.endswith(suffix))


def find_npy_filenames(path_to_dir, suffix=".npy"):
    """Names of NumPy files in a directory, sorted reproducibly."""
    from pathlib import Path
    return sorted(p.name for p in Path(path_to_dir).iterdir()
                  if p.is_file() and p.name.endswith(suffix))


def RMSE(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))


def filter_frames(pdb_file, xtc_file, threshold=0.001, chunk=1000):
    """Return 0-based frame indices without overlapping atoms. Distances in nm.

    Checks all atom pairs (including hydrogens).
    Uses a KD-tree to check the minimum distance without allocating an N^2
    distance matrix. Streams chunks rather than loading the trajectory at once.
    """
    import mdtraj as md
    from scipy.spatial import cKDTree
    if threshold < 0:
        raise ValueError('threshold must be nonnegative (nm)')
    ok_frames = []
    frame_offset = 0
    for traj in md.iterload(str(xtc_file), top=str(pdb_file), chunk=chunk):
        for i, xyz in enumerate(traj.xyz):
            min_dist = cKDTree(xyz).query(xyz, k=2)[0][:, 1].min()
            if min_dist > threshold:
                ok_frames.append(frame_offset + i)
        frame_offset += traj.n_frames
    return np.asarray(ok_frames, dtype=int)
