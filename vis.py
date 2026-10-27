"""Visualize TT-Occ occupancy predictions.

pred: a .pth file, or a folder holding .pth files (a scene folder, its Occ/ folder,
or a results root containing scene-*/Occ/).

  python vis.py results-Occ3D/Camera_OpenSeeD_VGGT_RAFT/scene-0003            # interactive
  python vis.py results-Occ3D/.../scene-0003/Occ/05.pth --save_dir out_vis    # save PNGs, no window
  python vis.py <pred> --data_path <nuscenes> --gt_path <Occ3D gts>           # lidar frame + GT (key G)

Keys (interactive): N next, B back, G toggle GT, S save PNG, Q quit.
"""
import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

COLORS = np.array([
    [200, 200, 200], [255, 120, 50], [255, 192, 203], [255, 255, 0], [0, 150, 245],
    [0, 255, 255], [255, 127, 0], [255, 0, 0], [255, 240, 150], [135, 60, 0],
    [160, 32, 240], [255, 0, 255], [175, 0, 75], [75, 0, 75], [150, 240, 80],
    [230, 230, 250], [0, 175, 0], [0, 0, 0],
], dtype=np.uint8)  # same as arguments.colors
colors_1 = COLORS / 255
empty_id = 17

PC_RANGE_EGO = np.array([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4])
PC_RANGE_LIDAR = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
VOXEL_SIZE = 0.2


def find_pred_files(pred):
    p = Path(pred)
    if p.is_file():
        return [p]
    for pattern in ("*.pth", "Occ/*.pth", "*/Occ/*.pth"):
        files = sorted(p.glob(pattern))
        if files:
            return files
    raise FileNotFoundError(f"no .pth files under {p}")


def ego_to_lidar(voxel_indices, voxel_cls, ego2lidar):
    pts = voxel_indices * VOXEL_SIZE + PC_RANGE_EGO[:3]
    pts = (ego2lidar @ np.concatenate([pts, np.ones((len(pts), 1))], 1).T).T[:, :3]
    keep = (pts >= PC_RANGE_LIDAR[:3]).all(1) & (pts < PC_RANGE_LIDAR[3:]).all(1)
    idx = np.floor((pts[keep] - PC_RANGE_LIDAR[:3]) / VOXEL_SIZE).astype(int)
    dims = np.round((PC_RANGE_LIDAR[3:] - PC_RANGE_LIDAR[:3]) / VOXEL_SIZE).astype(int)
    dense = np.full(tuple(dims), empty_id, dtype=int)
    dense[idx[:, 0], idx[:, 1], idx[:, 2]] = voxel_cls[keep]
    return dense


def dense_to_sparse(dense):
    idx = np.stack(np.nonzero(dense != empty_id), axis=-1)
    return idx, dense[idx[:, 0], idx[:, 1], idx[:, 2]]


def drop_empty(idx, cls):
    keep = (cls != 0) & (cls != empty_id)
    return idx[keep], cls[keep]


def to_voxel_grid(idx, cls):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(idx + 0.5)
    pcd.colors = o3d.utility.Vector3dVector(colors_1[cls])
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)


def load_frame(path, args, mapping):
    """Return (name, pred_grid, gt_grid or None)."""
    ckpt = torch.load(path, map_location="cpu")
    idx = ckpt["voxel_indices"].cpu().numpy()
    cls = ckpt["voxel_cls"].cpu().numpy()
    scene = path.parent.parent.name if path.parent.name == "Occ" else path.parent.name
    step = int(path.stem.split("_")[-1])
    name = f"{scene}_{step:02d}"
    gt = None

    if args.data_path:
        ego2world = np.loadtxt(Path(args.data_path) / scene / "ego" / f"{step:02d}.txt")
        lidar2world = np.loadtxt(Path(args.data_path) / scene / "lidar" / f"{step:02d}.txt")
        ego2lidar = np.linalg.inv(lidar2world) @ ego2world
        idx, cls = dense_to_sparse(ego_to_lidar(idx, cls, ego2lidar))
        if args.gt_path:
            token = mapping[scene]["sample_token"][f"{step:02d}"]
            data = np.load(Path(args.gt_path) / scene / token / "labels.npz")
            gt = to_voxel_grid(*drop_empty(*dense_to_sparse(data["semantics"].astype(int))))

    return name, to_voxel_grid(*drop_empty(idx, cls)), gt


def save_all(files, args, mapping):
    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=args.width, height=args.height, visible=False)
    vis.get_render_option().background_color = np.array([1, 1, 1])
    for f in files:
        name, pred, gt = load_frame(f, args, mapping)
        for tag, grid in (("pred", pred), ("gt", gt)):
            if grid is None:
                continue
            vis.clear_geometries()
            vis.add_geometry(grid)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(str(out / f"{name}_{tag}.png"), do_render=True)
        print("saved", name)
    vis.destroy_window()


def show(files, args, mapping):
    state = {"i": -1, "gt": False, "frame": None}
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=args.width, height=args.height)
    vis.get_render_option().background_color = np.array([1, 1, 1])

    def draw(first=False):
        name, pred, gt = state["frame"]
        grid = gt if (state["gt"] and gt is not None) else pred
        cam = None if first else vis.get_view_control().convert_to_pinhole_camera_parameters()
        vis.clear_geometries()
        vis.add_geometry(grid, reset_bounding_box=first)
        if cam is not None:
            vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
        print(f"[{state['i'] + 1}/{len(files)}] {name} ({'GT' if grid is gt else 'pred'})")

    def go(step):
        def cb(_):
            i = state["i"] + step
            if 0 <= i < len(files):
                first = state["i"] < 0
                state["i"], state["frame"] = i, load_frame(files[i], args, mapping)
                draw(first)
            return False
        return cb

    def toggle_gt(_):
        state["gt"] = not state["gt"]
        draw()
        return False

    def save(_):
        os.makedirs(args.save_dir or ".", exist_ok=True)
        path = os.path.join(args.save_dir or ".", f"{state['frame'][0]}.png")
        vis.capture_screen_image(path)
        print("saved", path)
        return False

    vis.register_key_callback(ord("N"), go(+1))
    vis.register_key_callback(ord("B"), go(-1))
    vis.register_key_callback(ord("G"), toggle_gt)
    vis.register_key_callback(ord("S"), save)
    vis.register_key_callback(ord("Q"), lambda v: v.close() or False)
    go(+1)(vis)
    vis.run()
    vis.destroy_window()


def main():
    ap = ArgumentParser(description="Visualize TT-Occ predictions")
    ap.add_argument("pred", help=".pth file or folder of .pth files")
    ap.add_argument("--data_path", help="extracted nuScenes folder (mapping.json, <scene>/ego, <scene>/lidar); enables lidar-frame view")
    ap.add_argument("--gt_path", help="Occ3D gts folder; needs --data_path; adds GT view")
    ap.add_argument("--save_dir", help="save PNGs here and do not open a window")
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--height", type=int, default=800)
    args = ap.parse_args()

    files = find_pred_files(args.pred)
    mapping = json.load(open(Path(args.data_path) / "mapping.json")) if args.gt_path else None
    if args.save_dir:
        save_all(files, args, mapping)
    else:
        show(files, args, mapping)


if __name__ == "__main__":
    main()
