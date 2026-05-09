from __future__ import annotations

import argparse
import os
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
from rembg import remove, new_session
import time

import cv2
import numpy as np



@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: List[float]


@dataclass
class ImageRec:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def run(cmd: List[str], cwd: Path | None = None) -> None:
    print("[cmd]", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def extract_frames(
    video_path: Path,
    image_dir: Path,
    every_n: int = 1,
    max_frames: int | None = None,
    remove_background: bool = False,
    background_color: Tuple[int, int, int] = (0, 0, 0),) -> int:
    image_dir.mkdir(parents=True, exist_ok=True)

    rembg_session = None
    if remove_background:
        print("[info] REMBG background removal enabled")
        rembg_session = new_session("u2net")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[info] video fps={fps:.3f}, size={width}x{height}, frames={total}")

    saved = 0
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % every_n == 0:
            out = image_dir / f"frame_{saved:06d}.png"

            if remove_background:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                cutout = remove(pil_img, session=rembg_session).convert("RGBA")
                arr = np.array(cutout)
                alpha = arr[..., 3:4].astype(np.float32) / 255.0
                fg = arr[..., :3].astype(np.float32)

                bg = np.zeros_like(fg, dtype=np.float32)
                bg[..., 0] = background_color[0]
                bg[..., 1] = background_color[1]
                bg[..., 2] = background_color[2]

                cv2.imwrite(str(out), frame)
                mask_dir = image_dir.parent / "masks"
                mask_dir.mkdir(exist_ok=True)
                alpha = np.array(cutout)[..., 3]
                mask = (alpha > 0).astype(np.uint8) * 255
                mask_path = mask_dir / f"frame_{saved:06d}.png"
                cv2.imwrite(str(mask_path), mask)
            else:
                cv2.imwrite(str(out), frame)

            saved += 1

            if max_frames is not None and saved >= max_frames:
                break

        idx += 1

    cap.release()
    print(f"[info] extracted {saved} frames")
    return saved


def run_colmap(
    colmap_bin: str,
    image_dir: Path,
    work_dir: Path,
    use_gpu: bool) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    dense_dir = work_dir / "dense"
    sparse_dir.mkdir(exist_ok=True)

    feat_cmd = [colmap_bin,
        "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", "1" if use_gpu else "0"]
    run(feat_cmd)

    matcher_cmd = [colmap_bin, "sequential_matcher", "--database_path", str(db_path)]
    matcher_cmd += ["--FeatureMatching.use_gpu", "1" if use_gpu else "0"]
    run(matcher_cmd)

    run([colmap_bin,
        "mapper",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir)])

    model0 = sparse_dir / "0"
    if not model0.exists():
        raise RuntimeError("COLMAP mapper did not produce sparse/0. Reconstruction failed.")

    run([colmap_bin,
        "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(model0),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP"])

    run([colmap_bin,
        "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.gpu_index", "0" if use_gpu else "-1"])

    run([colmap_bin,
        "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(dense_dir / "fused.ply")])


# based on colmap text model layout
def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*z*x + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*z*x - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def read_cameras_text(path: Path) -> Dict[int, Camera]:
    cams: Dict[int, Camera] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            cam_id = int(toks[0])
            cams[cam_id] = Camera(
                camera_id=cam_id,
                model=toks[1],
                width=int(toks[2]),
                height=int(toks[3]),
                params=[float(x) for x in toks[4:]],
            )
    return cams


def read_images_text(path: Path) -> Dict[int, ImageRec]:
    imgs: Dict[int, ImageRec] = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith("#"):
            i += 1
            continue
        toks = lines[i].split()
        image_id = int(toks[0])
        qvec = np.array([float(x) for x in toks[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in toks[5:8]], dtype=np.float64)
        camera_id = int(toks[8])
        name = toks[9]
        imgs[image_id] = ImageRec(image_id, qvec, tvec, camera_id, name)
        i += 2  # skip points2D line
    return imgs


def ensure_text_model(colmap_bin: str, sparse_model_dir: Path) -> None:
    if (sparse_model_dir / "cameras.txt").exists() and (sparse_model_dir / "images.txt").exists():
        return
    run([
        colmap_bin,
        "model_converter",
        "--input_path", str(sparse_model_dir),
        "--output_path", str(sparse_model_dir),
        "--output_type", "TXT",
    ])


def camera_to_intrinsics_dict(cam: Camera) -> Dict[str, float]:
    p = cam.params
    if cam.model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        k1 = k2 = p1 = p2 = 0.0
    elif cam.model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        fx = fy = f
        k1 = k2 = p1 = p2 = 0.0
    elif cam.model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = p[:4]
        fx = fy = f
        k2 = p1 = p2 = 0.0
    elif cam.model == "RADIAL":
        f, cx, cy, k1, k2 = p[:5]
        fx = fy = f
        p1 = p2 = 0.0
    elif cam.model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
    else:
        raise NotImplementedError(f"Unsupported COLMAP camera model for export: {cam.model}")
    return {
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "w": cam.width,
        "h": cam.height,
    }


def colmap_image_to_c2w(img: ImageRec) -> np.ndarray:
    r = qvec_to_rotmat(img.qvec)
    t = img.tvec.reshape(3, 1)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = r
    w2c[:3, 3:] = t
    c2w = np.linalg.inv(w2c)
    # convert COLMAP/OpenCV camera coords to Nerfstudio/OpenGL-ish convention
    convert = np.diag([1, -1, -1, 1]).astype(np.float64)
    c2w = c2w @ convert
    return c2w


def make_filtered_images(
    image_dir: Path,
    filtered_dir: Path,
    mask_dir: Path) -> None:

    filtered_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print("[info] generating REMBG-filtered images for Nerfstudio")
    session = new_session("u2net")

    for img_path in sorted(image_dir.glob("*.png")):
        pil_img = Image.open(img_path).convert("RGB")
        cutout = remove(pil_img, session=session).convert("RGBA")

        arr = np.array(cutout)
        cutout.save(filtered_dir / img_path.name)

        mask = (arr[..., 3] > 0).astype(np.uint8) * 255
        Image.fromarray(mask).save(mask_dir / img_path.name)

    print(f"[info] wrote filtered images to {filtered_dir}")
    print(f"[info] wrote masks to {mask_dir}")


def write_transforms_json(
    image_dir: Path,
    sparse_model_dir: Path,
    out_path: Path,
    mask_dir: Path | None = None,
) -> None:
    cams = read_cameras_text(sparse_model_dir / "cameras.txt")
    imgs = read_images_text(sparse_model_dir / "images.txt")
    if not imgs:
        raise RuntimeError("No registered images found in COLMAP sparse model.")

    first_cam = cams[next(iter(imgs.values())).camera_id]
    base = camera_to_intrinsics_dict(first_cam)

    frames = []
    for _, img in sorted(imgs.items(), key=lambda kv: kv[1].name):
        c2w = colmap_image_to_c2w(img)

        frame = {
            "file_path": f"../{image_dir.name}/{img.name}",
            "transform_matrix": c2w.tolist(),
        }

        if mask_dir is not None:
            frame["mask_path"] = f"../{mask_dir.name}/{img.name}"

        frames.append(frame)

    out = {
        **base,
        "camera_model": first_cam.model,
        "ply_file_path": "fused.ply",
        "frames": frames
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[info] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True, help="Input video file.")
    ap.add_argument("--output", type=Path, required=True, help="Output workspace directory.")
    ap.add_argument("--colmap-bin", default="colmap\\COLMAP.bat", help="COLMAP executable name or full path.")
    ap.add_argument("--every-n", type=int, default=10, help="Extract every Nth frame.")
    ap.add_argument("--max-frames", type=int, default=None, help="Optional cap on extracted frames.")
    ap.add_argument("--cpu", action="store_true", default=True, help="Disable GPU for SIFT and PatchMatch.")
    ap.add_argument("--remove-background", action="store_true", help="Use REMBG to remove background from extracted frames.")
    args = ap.parse_args()

    out = args.output.resolve()
    images = out / "images"
    colmap_dir = out / "colmap"
    ns_dir = out / "nerfstudio"
    ns_dir.mkdir(parents=True, exist_ok=True)
    if images.exists():
        shutil.rmtree(images)
    extracted = extract_frames(
        args.video.resolve(),
        images,
        every_n=args.every_n,
        max_frames=args.max_frames,
        remove_background=args.remove_background)
    if extracted < 2:
        raise RuntimeError("Need at least 2 frames for reconstruction.")

    run_colmap(colmap_bin=args.colmap_bin,
               image_dir=images,
               work_dir=colmap_dir,
               use_gpu=not args.cpu)

    sparse_model = colmap_dir / "sparse" / "0"
    ensure_text_model(args.colmap_bin, sparse_model)
    if args.remove_background:
        filtered_images = out / "images_filtered"
        masks = out / "masks"
        make_filtered_images(
            image_dir=images,
            filtered_dir=filtered_images,
            mask_dir=masks)
        write_transforms_json(
            filtered_images,
            sparse_model,
            ns_dir / "transforms.json",
            mask_dir=masks)
    else:
        write_transforms_json(images, sparse_model, ns_dir / "transforms.json")

    fused_src = colmap_dir / "dense" / "fused.ply"
    fused_dst = ns_dir / "fused.ply"
    while not fused_src.exists():
        time.sleep(20)
    shutil.copy2(fused_src, fused_dst)
    print(f"[info] copied dense point cloud to {fused_dst}")

    print("[done] Reconstruction complete.")
    print(f"[done] Sparse model: {sparse_model}")
    print(f"[done] Dense point cloud: {fused_src}")
    print(f"[done] Nerfstudio export: {ns_dir / 'transforms.json'}")

    print("[info] Starting Nerfstudio Splatfacto training...")
    experiment_dir = out / "nerfstudio_output" / "training"
    train_cmd = ["python",
        "-m",
        "nerfstudio.scripts.train",
        "splatfacto",
        "--data", str(ns_dir),
        "--output-dir", str(experiment_dir),
        "--vis", "tensorboard",
        "--viewer.quit-on-train-completion", "True",
        ]
    run(train_cmd)
    print("[done] Training complete.")
    
    print("[info] Exporting Gaussian splat to PLY...")
    outputs_dir = out / "nerfstudio_output\\training"
    if not outputs_dir.exists():
        raise RuntimeError("Nerfstudio outputs directory not found.")
    latest_config = sorted(
        outputs_dir.rglob("config.yml"),
        key=lambda p: p.stat().st_mtime,
    )[-1]

    export_cmd = ["python",
        "-m",
        "nerfstudio.scripts.exporter",
        "gaussian-splat",
        "--load-config",
        str(latest_config),
        "--output-dir",
        str(out / "nerfstudio_output" / "gaussian_export")]
    run(export_cmd)
    print("[info] Gaussian splat exported.")
    ply_files = list((out / "nerfstudio_output" / "gaussian_export").glob("*.ply"))
    if ply_files:
        print(f"[done] Final splat PLY: {ply_files[0]}")
    else:
        raise RuntimeError("No PLY file found after export.")
    mesh_cmd = [
        "python",
        "nerfstudio_to_unity_mesh.py",
        "--input", str(ply_files[0]),
        "--output", str(out / "nerfstudio_output" / "mesh_export" / "mesh.obj")]
    run(mesh_cmd)



if __name__ == "__main__":
    main()
