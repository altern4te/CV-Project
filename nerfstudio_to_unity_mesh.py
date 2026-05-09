import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sh0_to_rgb(f_dc):
    # Same basic SH DC conversion used in 3DGS viewers.
    C0 = 0.28209479177387814
    return np.clip(f_dc * C0 + 0.5, 0.0, 1.0)


def load_gaussian_splat_as_pointcloud(path: Path, opacity_thresh: float = 0.01):
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = v.dtype.names

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

    keep = np.ones(len(xyz), dtype=bool)

    if "opacity" in names:
        opacity = sigmoid(np.asarray(v["opacity"], dtype=np.float64))
        keep &= opacity >= opacity_thresh

    xyz = xyz[keep]

    colors = None
    if all(k in names for k in ["f_dc_0", "f_dc_1", "f_dc_2"]):
        f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float64)
        colors = sh0_to_rgb(f_dc)[keep]
    elif all(k in names for k in ["red", "green", "blue"]):
        colors = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float64)[keep] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def keep_largest_cluster(pcd, eps=0.02, min_points=100):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    
    if labels.max() < 0:
        print("[warn] no clusters found")
        return pcd

    largest = np.bincount(labels[labels >= 0]).argmax()
    keep = labels == largest

    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[keep])
    filtered.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[keep])

    print(f"[info] kept cluster {largest} with {keep.sum()} points")
    return filtered

def keep_near_center(pcd, radius=0.5):
    pts = np.asarray(pcd.points)
    center = pts.mean(axis=0)
    dist = np.linalg.norm(pts - center, axis=1)
    keep = dist < radius

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[keep])
    out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[keep])

    return out

def clean_mesh(mesh):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="Gaussian splat .ply from ns-export gaussian-splat")
    ap.add_argument("--output", required=True, type=Path, help="Output mesh .obj/.ply")
    ap.add_argument("--opacity-thresh", type=float, default=0.03)
    ap.add_argument("--voxel-size", type=float, default=0.01)
    ap.add_argument("--poisson-depth", type=int, default=15)
    ap.add_argument("--density-quantile", type=float, default=0.08)
    ap.add_argument("--simplify", type=int, default=200000)
    ap.add_argument("--remove-small-clusters", action="store_true", help="For object generation rather than scenes, removes clumps of points from imperfect filtering of images")
    ap.add_argument("--smoothing", type=int, default=0, help="Recommended ~40 iterations")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("[info] loading gaussian splat as point cloud")
    pcd = load_gaussian_splat_as_pointcloud(args.input, args.opacity_thresh)

    if pcd.is_empty():
        raise RuntimeError("Point cloud is empty after opacity filtering. Lower --opacity-thresh.")

    print(f"[info] points after filtering: {len(pcd.points)}")

    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(args.voxel_size)
        print(f"[info] points after voxel downsample: {len(pcd.points)}")

    pcd = keep_largest_cluster(pcd, eps=0.35, min_points=1000)
    print("[info] estimating normals")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(args.voxel_size * 5, 0.03),
            max_nn=100,
        )
    )

    try:
        pcd.orient_normals_consistent_tangent_plane(50)
    except Exception:
        print("[warn] normal orientation failed; continuing anyway")

    print("[info] running poisson reconstruction")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=args.poisson_depth,
    )
    
    densities = np.asarray(densities)
    density_thresh = np.quantile(densities, args.density_quantile)
    mesh.remove_vertices_by_mask(densities < density_thresh)

    if args.remove_small_clusters:
        print("[info] removing small clusters")
        labels = np.array(mesh.cluster_connected_triangles()[0])
        largest_cluster = np.bincount(labels).argmax()
        mesh.remove_triangles_by_mask(labels != largest_cluster)
        mesh.remove_unreferenced_vertices()

    if args.smoothing > 0:
        print("[info] smoothing")
        mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smoothing)

    mesh.compute_vertex_normals()
    mesh = clean_mesh(mesh)

    if args.simplify and args.simplify > 0:
        print(f"[info] simplifying to ~{args.simplify} triangles")
        mesh = mesh.simplify_quadric_decimation(args.simplify)
        mesh = clean_mesh(mesh)

    print(f"[info] writing {args.output}")
    o3d.io.write_triangle_mesh(str(args.output), mesh)
    print("[done]")


if __name__ == "__main__":
    main()
