# Video to COLMAP to Nerf to Mesh pipeline

This script extracts frames from a video, runs COLMAP end to end, and exports:
- sparse reconstruction under `output/colmap/sparse/0`
- dense fused point cloud at `output/colmap/dense/fused.ply`
- Poisson mesh at `output/colmap/dense/meshed-poisson.ply`
- Nerfstudio-style `transforms.json` at `output/nerfstudio/transforms.json`

Then runs nerfstudio and exports:
- training outputs under `output/nerfstudio_output/training`
- the exported splat at `output/nerfstudio_output/gaussian_export`
- the final mesh at `output/nerfstudio_output/mesh_export`

## Requirements
- COLMAP installed and available on PATH, or pass `--colmap-bin`
- Python packages: `opencv-python`, `numpy`, `pillow`, `rembg`, `open3d`, `pltfile`, `onnxruntime`, `torch`, `torchvision`, `nerfstudio`
- Microsoft Visual Studio 2022 C++ Compiler (for Nerfstudio)

## Example

```bash
python colmap_nerf_mesh_pipeline.py \
--video video.mp4 \
--output output \
--every-n 10 \
--colmap-bin "[PATH\colmap\COLMAP.bat]"
```


## Notes
- For GPU usage on pytorch specific versions are needed depending on GPU, be wary and make sure the correct one is installed. For example, RTX 5060 requires cu128+ be installed.
- Nerfstudio may not recognize the correct architecture in place. Check system environmental variables to make sure your desired CUDA version is the highest on the list, move if not. If that does not work you may need to specify your version with the following command. 
```bash
set TORCH_CUDA_ARCH_LIST=[VERSION] 
```
- gsplat may not be able to handle MAX_JOBS=10, the default, if the program crashes when gsplat is being set up, run the command
```bash
set MAX_JOBS=1
```
- This program requires the Microsoft Visual Studio 2022 C++ Compiler. If this isnt the primary compiler on your system run it on the Developer Command Prompt for MSVS 2022.
- The mesh from the pipeline almost certainly will not be tuned correctly and will need to be reran depending on your scene and expectations. If you experience shells around detailed objects, decrease voxel size or density. If you notice artifacts around your background removed object, use remove small clusters. Experimentation will be your best bet in getting a good mesh, and further correction with tools such as Blender may be required to clean the mesh.