# 3D-Object-Reconstruction-SfM

# 3D Object Reconstruction Pipeline

This repository contains a **Structure‑from‑Motion (SfM)** and **surface‑reconstruction** pipeline that converts a set of 2‑D photographs into a colored 3‑D mesh.  
The reconstructed model can be exported as **PLY** or imported into simulators such as **Blender**, **MeshLab**, or **Gazebo** for robotic manipulation.

---

## Stages of Structure from Motion
| Stage | Technique | Implementation |
|-------|-----------|----------------|
| Camera Calibration | Chessboard (9 × 6) corner detection | `test.py` §1 |
| Feature Detection | SIFT keypoints & descriptors | `test.py` §2 |
| Feature Matching | BF‑Matcher + Lowe ratio + RANSAC F‑matrix | `test.py` §3 |
| View‑Graph Pruning | Triplet cycle‑consistency check | `test.py` §4 |
| Pose Recovery | Essential matrix → `cv2.recoverPose` | `test.py` §6 |
| Triangulation | `cv2.triangulatePoints` | `test.py` §7 |
| Point‑Cloud Colouring | Per‑pixel RGB sampling | `test.py` §9 |
| Mesh Generation | Poisson surface reconstruction (Open3D) | `test.py` §10 |
| Interactive Demo | Jupyter notebook | `python_script_reconstruction.ipynb` |

---

## Quick Start

```bash
# Clone and enter the repo
git clone https://github.com/your‑username/3d-reconstruction.git
cd 3d-reconstruction
```
```bash
# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```
```bash
# Install dependencies
pip install -r requirements.txt
```
```bash
# Run the notebook or script
jupyter notebook python_script_reconstruction.ipynb   # interactive
```
or
```bash
python test.py                                        # terminal
```

### Output
- `reconstruction_colored.ply` – sparse, coloured point cloud  
- `mesh_poisson_trimmed.ply` – cleaned Poisson mesh  
- Matplotlib window showing camera trajectory + point cloud  

---

## Repository Layout
```
.
├── dataset/
│   ├── chessboard/        # calibration images
│   └── sippu/             # object images for SfM
├── python_script_reconstruction.ipynb
├── test.py                # stand‑alone pipeline
├── requirements.txt
└── README.md              # (this file)
```
> **Tip:** adjust the hard‑coded dataset paths at the top of `test.py`.

---

## Methodology

1. **Calibration** – OpenCV chessboard routine yields intrinsic matrix **K** & distortion coefficients.  
2. **SIFT + BFMatcher** – robust feature detection; Lowe’s ratio = 0.75.  
3. **Fundamental Matrix (RANSAC)** – removes outliers; edges stored in a view‑graph.  
4. **Cycle Consistency** – prunes geometrically inconsistent triplets.  
5. **Essential Matrix & Poses** – recover **R, t** pairs; propagate via BFS.  
6. **Triangulation** – back‑project inlier correspondences into 3‑D space.  
7. **Surface Reconstruction** – Poisson meshing + density trimming (Open3D).  


---

## Troubleshooting

| Symptom | Possible Fix |
|---------|--------------|
| Few keypoints detected | Increase lighting / texture or image resolution |
| Pose init fails | Ensure first two images have >500 inliers & 50 % overlap |
| Noisy point cloud | Capture more views, tighten RANSAC threshold, bundle‑adjust |
| Poisson meshing slow | Lower `depth` parameter (e.g. 10 → 8) |

---



## Author

**Kriti Chauhan** – k24078085@kcl.ac.uk  
Please open an issue for questions or improvements.
