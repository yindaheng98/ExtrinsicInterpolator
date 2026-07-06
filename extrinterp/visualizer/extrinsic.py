from typing import List, Sequence

import open3d as o3d
import torch

from ..abc import Extrinsic


def plot_extrinsic(
    extrinsic: Extrinsic,
    geometries: List,
    axis_length: float = 1.0,
):
    """Add one extrinsic coordinate frame to an Open3D geometry list."""
    transform = torch.eye(4, dtype=torch.float64)
    transform[:3, :3] = extrinsic.R.detach().cpu().to(dtype=torch.float64)
    transform[:3, 3] = extrinsic.T.detach().cpu().reshape(-1)[:3].to(dtype=torch.float64)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length)
    frame.transform(transform.numpy())
    geometries.append(frame)

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
    geometries.append(body)

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector([
        [0.0, 0.0, 0.0],
        [-axis_length * 0.6, -axis_length * 0.4, axis_length],
        [axis_length * 0.6, -axis_length * 0.4, axis_length],
        [axis_length * 0.6, axis_length * 0.4, axis_length],
        [-axis_length * 0.6, axis_length * 0.4, axis_length],
    ])
    lines.lines = o3d.utility.Vector2iVector([
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
    ])
    lines.colors = o3d.utility.Vector3dVector([[0.0, 0.0, 0.0] for _ in range(8)])
    lines.transform(transform.numpy())
    geometries.append(lines)
    return geometries


def plot_extrinsics(
    extrinsics: Sequence[Extrinsic],
):
    """Show multiple extrinsics and their translation trajectory with Open3D."""
    extrinsics = list(extrinsics)
    positions = torch.stack([
        extrinsic.T.detach().cpu().reshape(-1)[:3]
        for extrinsic in extrinsics
    ]).to(dtype=torch.float64)
    ranges = positions.max(dim=0).values - positions.min(dim=0).values
    axis_length = max(ranges.max().item(), 1.0) * 0.08
    geometries = []

    trajectory = o3d.geometry.LineSet()
    trajectory.points = o3d.utility.Vector3dVector(positions.numpy())
    lines = [[idx, idx + 1] for idx in range(len(extrinsics) - 1)]
    trajectory.lines = o3d.utility.Vector2iVector(lines)
    trajectory.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for _ in lines])
    geometries.append(trajectory)
    for extrinsic in extrinsics:
        plot_extrinsic(
            extrinsic,
            geometries,
            axis_length=axis_length,
        )

    o3d.visualization.draw_geometries(geometries)
    return geometries
