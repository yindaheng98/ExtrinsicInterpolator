from typing import List, Sequence, Tuple

import open3d as o3d
import open3d.visualization.gui as gui
import torch

from ..abc import Extrinsic


def plot_extrinsic(
    extrinsic: Extrinsic,
    axis_length: float = 1.0,
):
    """Build one extrinsic coordinate frame and camera marker."""
    transform = torch.eye(4, dtype=torch.float64)
    transform[:3, :3] = extrinsic.R.detach().cpu().to(dtype=torch.float64)
    transform[:3, 3] = extrinsic.T.detach().cpu().reshape(-1)[:3].to(dtype=torch.float64)

    mesh = o3d.geometry.TriangleMesh()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length)
    frame.transform(transform.numpy())
    mesh += frame

    body = o3d.geometry.TriangleMesh()
    body.vertices = o3d.utility.Vector3dVector([
        [-axis_length * 0.6, -axis_length * 0.4, axis_length],
        [axis_length * 0.6, -axis_length * 0.4, axis_length],
        [axis_length * 0.6, axis_length * 0.4, axis_length],
        [-axis_length * 0.6, axis_length * 0.4, axis_length],
    ])
    body.triangles = o3d.utility.Vector3iVector([
        [0, 3, 2],
        [0, 2, 1],
        [2, 3, 0],
        [1, 2, 0],
    ])
    body.paint_uniform_color([1.0, 0.65, 0.0])
    body.compute_vertex_normals()
    body.transform(transform.numpy())
    mesh += body

    marker_lines = o3d.geometry.LineSet()
    marker_lines.points = o3d.utility.Vector3dVector([
        [0.0, 0.0, 0.0],
        [-axis_length * 0.6, -axis_length * 0.4, axis_length],
        [axis_length * 0.6, -axis_length * 0.4, axis_length],
        [axis_length * 0.6, axis_length * 0.4, axis_length],
        [-axis_length * 0.6, axis_length * 0.4, axis_length],
    ])
    marker_lines.lines = o3d.utility.Vector2iVector([
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
    ])
    marker_lines.colors = o3d.utility.Vector3dVector([[0.0, 0.0, 0.0] for _ in range(8)])
    marker_lines.transform(transform.numpy())
    return mesh, marker_lines


def plot_extrinsics(
    extrinsics: Sequence[Extrinsic],
    size_scale: float = 1.0,
    quantile: float = 0.25,
    clamp_min_scale: float = 0.01,
    line_color: Sequence[float] = (0.5, 0.5, 0.5),
    name: str = "extrinsics",
):
    """Build Open3D geometries for multiple extrinsics and their trajectory."""
    extrinsics = list(extrinsics)
    positions = torch.stack([
        extrinsic.T.detach().cpu().reshape(-1)[:3]
        for extrinsic in extrinsics
    ]).to(dtype=torch.float64)
    ranges = positions.max(dim=0).values - positions.min(dim=0).values
    max_range = max(ranges.max().item(), 1.0)
    spacing = (positions[1:] - positions[:-1]).norm(dim=1)
    spacing = torch.concat((spacing, torch.tensor([max_range], dtype=positions.dtype)))
    axis_length = torch.quantile(spacing, quantile).clamp_min(max_range * clamp_min_scale).item() * size_scale
    pose_mesh = o3d.geometry.TriangleMesh()
    pose_lines = o3d.geometry.LineSet()

    trajectory = o3d.geometry.LineSet()
    trajectory.points = o3d.utility.Vector3dVector(positions.numpy())
    lines = [[idx, idx + 1] for idx in range(len(extrinsics) - 1)]
    trajectory.lines = o3d.utility.Vector2iVector(lines)
    trajectory.colors = o3d.utility.Vector3dVector([line_color for _ in lines])
    for extrinsic in extrinsics:
        mesh, marker_lines = plot_extrinsic(extrinsic, axis_length=axis_length)
        pose_mesh += mesh
        pose_lines += marker_lines

    return [
        (f"{name} trajectory", trajectory),
        (f"{name} camera poses", pose_mesh),
        (f"{name} camera pose lines", pose_lines),
    ]


def draw_geometries(
    geometries: Sequence[Tuple[str, o3d.geometry.Geometry]],
    title: str = "Extrinsic Visualizer",
) -> None:
    app = gui.Application.instance
    app.initialize()
    visualizer = o3d.visualization.O3DVisualizer(title, 1024, 768)
    visualizer.show_settings = True
    for name, geometry in geometries:
        visualizer.add_geometry(name, geometry)
    visualizer.reset_camera_to_default()
    app.add_window(visualizer)
    app.run()
