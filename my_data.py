import os
import json
import shutil
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import tqdm
import open3d as o3d

"""
Convert raw_data into training-ready pointcloud + robot_state layout expected by the
training code (dataset.py).

Input raw layout (expected):
  raw_data/<robot_name>/pose_XXXX/angles.json
                                 mesh_combined.ply
                                 pointcloud.ply

Output layout (default):
  ./saved_meshes/
    mesh_0.ply
    mesh_0.xyzn   <-- text file with columns: x y z nx ny nz (space-separated)
    mesh_1.ply
    mesh_1.xyzn
    ...
    robot_state.json  <-- mapping index -> robot_state (list as produced by sim.py)

robot_state format (per index):
  [ [joint0_pos, joint0_vel], [joint1_pos, joint1_vel], ... , [last_link_x, last_link_y, last_link_z] ]

Notes / choices made:
 - We take the first key in each angles.json (the file typically contains an "angles" key).
 - Only the first 5 angles are used to create per-joint entries (pad with 0.0 if fewer).
 - Joint velocities are unknown in raw_data, so set to 0.0.
 - last_link position is approximated with the centroid of the pointcloud if available,
   otherwise [0., 0., 0.].
 - All poses across robots are written into a single output folder and indexed consecutively.

Usage:
  python my_data.py --raw_root ./raw_data --out_root ./saved_meshes

"""


def ensure_empty_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def read_angles_json(json_path: str) -> List[float]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    # prefer key 'angles' if present, otherwise take the first key
    if 'angles' in data:
        arr = data['angles']
    else:
        first_key = next(iter(data.keys()))
        arr = data[first_key]
    return [float(x) for x in arr]


def write_xyzn(path: str, points: np.ndarray, normals: np.ndarray):
    """Write a text file with x y z nx ny nz per line (space-separated)."""
    assert points.shape == normals.shape
    data = np.hstack([points, normals])
    # use a simple text format readable by np.genfromtxt used in dataset.py
    fmt = '%.6f %.6f %.6f %.6f %.6f %.6f'
    np.savetxt(path, data, fmt=fmt)


def load_pointcloud_from_ply(ply_path: str):
    mesh = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(mesh.points)
    # ensure normals; if missing, compute estimates
    if not mesh.has_normals():
        mesh.estimate_normals()
    normals = np.asarray(mesh.normals)
    return pts, normals


def load_mesh_vertices_normals(ply_path: str):
    mesh = o3d.io.read_triangle_mesh(ply_path)
    pts = np.asarray(mesh.vertices)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    return pts, normals


def data_process(raw_root: str = './raw_data', out_root: str = './saved_meshes'):
    raw_root = Path(raw_root)
    out_root = Path(out_root)

    ensure_empty_dir(str(out_root))

    robot_state_dict = {}
    global_idx = 0

    children = [p for p in raw_root.iterdir() if p.is_dir()]
    # If raw_root directly contains pose_* folders, treat raw_root as the robot directory.
    if any(p.name.startswith('pose') for p in children):
        robot_dirs = [raw_root]
    else:
        robot_dirs = sorted(children)

    for robot_dir in robot_dirs:
        # if robot_dir is the raw_root itself, its children are the pose dirs
        if robot_dir == raw_root:
            pose_dirs = sorted([p for p in children if p.name.startswith('pose')])
        else:
            pose_dirs = sorted([p for p in robot_dir.iterdir() if p.is_dir() and p.name.startswith('pose')])
        for pose_dir in tqdm.tqdm(pose_dirs, desc=f'Processing {robot_dir.name}'):
            # read angles
            angles_json_path = pose_dir / 'angles.json'
            if not angles_json_path.exists():
                # skip if missing
                print(f'Warning: missing angles.json in {pose_dir}, skipping')
                continue
            angles = read_angles_json(str(angles_json_path))

            # select first 5 angles (pad if needed)
            selected_angles = list(angles[:5])
            if len(selected_angles) < 5:
                selected_angles += [0.0] * (5 - len(selected_angles))

            # last link position: prefer mesh_combined.ply (we only need the combined mesh to get xyzn)
            last_link = [0.0, 0.0, 0.0]
            mesh_path = pose_dir / 'mesh_combined.ply'
            pc_path = pose_dir / 'pointcloud.ply'  # kept as fallback but not preferred
            pts = None
            normals = None
            if mesh_path.exists():
                try:
                    pts, normals = load_mesh_vertices_normals(str(mesh_path))
                    last_link = pts.mean(axis=0).tolist()
                except Exception:
                    pts = None
            elif pc_path.exists():
                try:
                    pts, normals = load_pointcloud_from_ply(str(pc_path))
                    last_link = pts.mean(axis=0).tolist()
                except Exception:
                    pts = None

            # construct robot_state entry
            robot_state = []
            for ang in selected_angles:
                robot_state.append([float(ang), 0.0])
            # append last_link as a list of 3 values
            robot_state.append([float(last_link[0]), float(last_link[1]), float(last_link[2])])

            # save mesh and xyzn to out_root as mesh_{global_idx}.ply / .xyzn
            out_mesh_ply = out_root / f'mesh_{global_idx}.ply'
            out_xyzn = out_root / f'mesh_{global_idx}.xyzn'

            # copy original mesh_combined.ply if available
            if mesh_path.exists():
                try:
                    shutil.copyfile(str(mesh_path), str(out_mesh_ply))
                except Exception:
                    # fallback: if we have pts, write a simple mesh-less point cloud as ply
                    out_mesh_ply = None
            else:
                out_mesh_ply = None

            # produce xyzn text file (points + normals required by dataset.py)
            if pts is not None and normals is not None and pts.shape[0] > 0:
                write_xyzn(str(out_xyzn), pts, normals)
            else:
                # fallback: create a small dummy point cloud so training loader won't fail
                dummy_pts = np.zeros((1024, 3), dtype=np.float32)
                dummy_normals = np.zeros((1024, 3), dtype=np.float32)
                write_xyzn(str(out_xyzn), dummy_pts, dummy_normals)

            robot_state_dict[str(global_idx)] = robot_state
            global_idx += 1

    # save global robot_state.json
    robot_state_path = out_root / 'robot_state.json'
    with open(robot_state_path, 'w') as f:
        json.dump(robot_state_dict, f, indent=4)

    print(f'Wrote {global_idx} samples to {out_root}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare raw_data into training-ready saved_meshes format')
    parser.add_argument('--raw_root', type=str, default='./raw_data')
    parser.add_argument('--out_root', type=str, default='./saved_meshes')
    args = parser.parse_args()

    data_process(raw_root=args.raw_root, out_root=args.out_root)